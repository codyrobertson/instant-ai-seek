// ias popup: shows whether the local model is ready (fully offline after that).

const statusEl = document.getElementById("status");
const detailEl = document.getElementById("detail");

(async () => {
  try {
    const resp = await chrome.runtime.sendMessage({ type: "ias:status" });
    if (resp?.ready) {
      statusEl.textContent = "model ready — fully offline";
      statusEl.className = "ok";
      detailEl.textContent = `Backbone: ${resp.backbone} (${resp.classes}-class). Detector loaded into memory.`;
    } else {
      statusEl.textContent = "model not loaded";
      statusEl.className = "err";
      detailEl.textContent = resp?.error || "Reopen this popup after the page finishes loading.";
    }
  } catch {
    statusEl.textContent = "unavailable";
    statusEl.className = "err";
  }
})();
