// ias background service worker.
// Routes inference requests to the offscreen document (which owns the ONNX
// session and does all image processing). Ensures the offscreen doc exists.
// Tracks diagnostic counters for status reporting.

const OFFSCREEN_URL = "offscreen.html";

const state = { ready: false, backbone: null, classes: null, ep: null, inferCalls: 0, inferResponses: 0, inferErrors: 0, lastError: null };

let offscreenReady = null;

async function ensureOffscreen() {
  if (offscreenReady) return offscreenReady;
  offscreenReady = (async () => {
    const has = await chrome.offscreen.hasDocument?.().catch(() => false);
    if (!has) {
      await chrome.offscreen.createDocument({
        url: OFFSCREEN_URL,
        reasons: ["DOM_SCRAPING"],
        justification: "Run local ONNX inference on page images (no data leaves the device).",
      });
    }
    return true;
  })();
  return offscreenReady;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type !== "ias:infer") return false;
  state.inferCalls += 1;
  ensureOffscreen()
    .then(async () => {
      const resp = await chrome.runtime.sendMessage({
        type: "ias:infer",
        dataUrl: msg.dataUrl,
        side: msg.side,
      });
      if (resp && resp.pAi !== undefined) {
        state.inferResponses += 1;
      } else {
        state.inferErrors += 1;
        state.lastError = resp?.error || "offscreen empty response";
      }
      sendResponse(resp ?? { error: "offscreen did not respond" });
    })
    .catch((e) => {
      state.inferErrors += 1;
      state.lastError = String(e?.message || e);
      sendResponse({ error: String(e?.message || e) });
    });
  return true;
});

// offscreen announces readiness once the model is in memory
chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.type === "ias:ready") {
    state.ready = true;
    state.backbone = msg.backbone ?? null;
    state.classes = msg.classes ?? null;
    state.ep = msg.ep ?? null;
  }
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type !== "ias:status") return false;
  sendResponse(state);
  return false;
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.runtime.sendMessage({ type: "ias:install" }).catch(() => {});
});
