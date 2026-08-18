// ias offscreen inference document.
// Loads the ONNX model once, then serves "ias:infer" messages.
// Preprocessing contract comes from the authoritative bundle manifest
// (detector/model_bundle_manifest.json) — never hard-code it independently.
// WebGPU uses the fp16 model (native fp16 compute); WASM falls back to q8.
// P(ai) = softmax(logits)[1].

let INPUT_SIZE = 384;
let RESIZE_TO = 416;
let MEAN = [0.485, 0.456, 0.406];
let STD = [0.229, 0.224, 0.225];

async function loadContract() {
  try {
    const resp = await fetch(chrome.runtime.getURL("lib/model_bundle_manifest.json"));
    const m = await resp.json();
    const c = m.contract;
    INPUT_SIZE = c.input_size;
    RESIZE_TO = c.resize_to;
    MEAN = c.mean;
    STD = c.std;
  } catch (e) {
    console.error("[ias] manifest unavailable, using defaults: " + (e?.message || e));
  }
}

let session = null;
let teacherSession = null;
let initPromise = null;
let epUsed = "wasm";
let modelIsFp16 = false;
const perfTimes = [];
let runChain = Promise.resolve(); // ORT sessions reject concurrent run()

// float32 -> float16 conversion (little-endian IEEE 754)
function toFloat16Array(src) {
  const out = new Uint16Array(src.length);
  const dv = new DataView(new ArrayBuffer(4));
  for (let i = 0; i < src.length; i++) {
    const f = src[i];
    if (f === 0) { out[i] = 0; continue; }
    dv.setFloat32(0, f, true);
    const bits = dv.getUint32(0, true);
    const sign = (bits >>> 16) & 0x8000;
    const exp = (bits >>> 23) & 0xff;
    const mant = bits & 0x7fffff;
    const e16 = exp - 127 + 15;
    if (e16 <= 0) out[i] = sign;
    else if (e16 >= 31) out[i] = sign | 0x7c00;
    else out[i] = sign | (e16 << 10) | (mant >> 13);
  }
  return out;
}

async function initOrt() {
  if (session) return session;
  if (!initPromise) {
    initPromise = (async () => {
      await loadContract();
      // forced EP override (E2E contract testing): "wasm" | "webgpu" | undefined
      let forcedEp = null;
      try {
        forcedEp = (await chrome.storage.local.get("ias_force_ep")).ias_force_ep || null;
      } catch (e) { /* storage unavailable — auto-detect */ }
      ort.env.wasm.wasmPaths = chrome.runtime.getURL("lib/onnxruntime-web/");
      const log = (msg) => console.error("[ias]", msg);
      const useGpu = forcedEp ? forcedEp === "webgpu" : typeof navigator !== "undefined" && !!navigator.gpu;
      const makeSession = async (name) => {
        const buf = await (await fetch(chrome.runtime.getURL("lib/" + name))).arrayBuffer();
        if (useGpu) {
          try {
            return await Promise.race([
              ort.InferenceSession.create(buf, { executionProviders: ["webgpu"] }),
              new Promise((_, rej) => setTimeout(() => rej(new Error("webgpu init timeout")), 8000)),
            ]);
          } catch (e) {
            log("webgpu create failed for " + name + ": " + (e?.message || e));
            const fb = await (await fetch(chrome.runtime.getURL("lib/model_cnn_q.onnx"))).arrayBuffer();
            return await ort.InferenceSession.create(fb, { executionProviders: ["wasm"] });
          }
        }
        return await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
      };
      const modelName = useGpu ? "model_cnn_fp16.onnx" : "model_cnn_q.onnx";
      const teacherName = useGpu ? "model_cnn_teacher_fp16.onnx" : "model_cnn_teacher_q.onnx";
      session = await makeSession(modelName);
      try {
        teacherSession = await makeSession(teacherName);
        log("teacher session loaded (ensemble)");
      } catch (e) {
        teacherSession = null;
        log("teacher unavailable, student only: " + (e?.message || e));
      }
      epUsed = useGpu && session ? "webgpu" : "wasm";
      const meta = session.inputMetadata && session.inputMetadata[0];
      modelIsFp16 = meta ? meta.type === "float16" : false;
      log("session ready, ep=" + epUsed + " fp16=" + modelIsFp16 + " ensemble=" + !!teacherSession);
      chrome.runtime.sendMessage({
        type: "ias:ready",
        backbone: "resnet18 (local ONNX)",
        classes: 2,
        ep: epUsed,
      }).catch(() => {});
      return session;
    })();
  }
  return initPromise;
}

async function decodeToTensor(dataUrl, side) {
  const img = await createImageBitmap(await (await fetch(dataUrl)).blob());
  const scale = RESIZE_TO / Math.min(img.width, img.height);
  const w = Math.max(1, Math.round(img.width * scale));
  const h = Math.max(1, Math.round(img.height * scale));
  // resize to (w,h) on an intermediate canvas, then center-crop 256 —
  // identical to the Python harness (resize shortest side -> crop)
  const resized = document.createElement("canvas");
  resized.width = w;
  resized.height = h;
  resized.getContext("2d", { willReadFrequently: true }).drawImage(img, 0, 0, w, h);
  img.close();
  const left = Math.floor((w - INPUT_SIZE) / 2);
  const top = Math.floor((h - INPUT_SIZE) / 2);
  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(resized, left, top, INPUT_SIZE, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const n = INPUT_SIZE * INPUT_SIZE;
  const out32 = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out32[i] = (data[i * 4] / 255 - MEAN[0]) / STD[0];
    out32[n + i] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    out32[2 * n + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  if (modelIsFp16) {
    return new ort.Tensor("float16", toFloat16Array(out32), [1, 3, INPUT_SIZE, INPUT_SIZE]);
  }
  return new ort.Tensor("float32", out32, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

async function infer(dataUrl, side) {
  const t0 = performance.now();
  const sess = await initOrt();
  const input = await decodeToTensor(dataUrl, side);
  const pAi = await new Promise((resolve, reject) => {
    runChain = runChain.then(async () => {
      const runSess = async (s) => {
        const results = await s.run({ image: input });
        let logits = results.logit.data; // [2] for 2-class head
        if (logits instanceof Uint16Array) {
        // fp16 output arrives as raw bit patterns — decode to floats
        const dv = new DataView(new ArrayBuffer(4));
        logits = Array.from(logits, (bits) => {
          const sign = (bits & 0x8000) ? -1 : 1;
          const exp = (bits >> 10) & 0x1f;
          const mant = bits & 0x3ff;
          if (exp === 0) return sign * mant * 2 ** -24;
          if (exp === 31) return mant ? NaN : sign * Infinity;
          return sign * (1 + mant / 1024) * 2 ** (exp - 15);
        });
      }
        const e0 = Math.exp(logits[0] - Math.max(logits[0], logits[1]));
        const e1 = Math.exp(logits[1] - Math.max(logits[0], logits[1]));
        return e1 / (e0 + e1);
      };
      const probs = [await runSess(sess)];
      if (teacherSession) probs.push(await runSess(teacherSession));
      resolve(Math.max(...probs));
    }).catch((e) => reject(e));
  });
  perfTimes.push(performance.now() - t0);
  if (perfTimes.length > 50) perfTimes.shift();
  // (perf samples are served on-demand via the ias:perf message)
  return pAi;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === "ias:perf") {
    sendResponse({ ep: epUsed, samples: perfTimes.slice(-30), n: perfTimes.length });
    return false;
  }
  return false;
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type !== "ias:infer") return false;
  infer(msg.dataUrl, msg.side)
    .then((pAi) => sendResponse({ pAi }))
    .catch((e) => sendResponse({ error: String(e?.message || e) }));
  return true; // async response
});
