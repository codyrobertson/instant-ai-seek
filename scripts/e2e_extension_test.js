// E2E: load the built extension in headless Chrome, open a page with eval
// images, collect badge scores, and compare against the Node harness
// (detector/detect.py) — the shipped extension must agree with the benchmark.
//
// Usage: node scripts/e2e_extension_test.js [--images n] [--ep=wasm|webgpu] [--hard-real n2]
//   --ep forces the extension's execution provider via chrome.storage
//   (the offscreen reads ias_force_ep); webgpu falls back to wasm in
//   headless environments, which is itself a valid parity check.
//   --hard-real n2 also serves n2 in-the-wild real photos (data/probe/real).
const { execFileSync } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");
const os = require("node:os");
const http = require("node:http");

const ROOT = path.resolve(__dirname, "..");
const EXT = path.join(ROOT, "extension");
const EVAL_DIR = path.join(ROOT, "data", "eval");
const PROBE_REAL_DIR = path.join(ROOT, "data", "probe", "real");

const N = parseInt(process.argv.find((a) => a.startsWith("--images="))?.split("=")[1] || "6", 10);
const EP = process.argv.find((a) => a.startsWith("--ep="))?.split("=")[1] || null;
const HARD_REAL = parseInt(process.argv.find((a) => a.startsWith("--hard-real="))?.split("=")[1] || "0", 10);

async function main() {
  // pick N eval AI images + N real
  const aiFiles = fs.readdirSync(path.join(ROOT, "data", "eval", "ai")).slice(0, N);
  const realFiles = fs.readdirSync(path.join(ROOT, "data", "eval", "real")).slice(0, N);
  const pairs = [
    ...aiFiles.map((f) => ["ai", f]),
    ...realFiles.map((f) => ["real", f]),
  ];
  if (HARD_REAL > 0) {
    if (!fs.existsSync(PROBE_REAL_DIR)) {
      console.log("hard-real packet skipped: data/probe/real not present (public export excludes it)");
    } else {
      pairs.push(...fs.readdirSync(PROBE_REAL_DIR).slice(0, HARD_REAL).map((f) => ["hardreal", f]));
    }
  }

  // Node harness reference scores
  const baseDir = (c) => (c === "hardreal" ? PROBE_REAL_DIR : path.join(EVAL_DIR, c));
  const paths = pairs.map(([c, f]) => path.join(baseDir(c), f));
  const out = execFileSync("python3", [path.join(ROOT, "detector", "detect.py"), "--batch"], {
    input: paths.join("\n"),
    encoding: "utf8",
  });
  const refs = out.split("\n").filter((l) => /^\d\.\d+/.test(l)).map(Number);
  // serve eval images + the test page over http (same-origin, avoids taint)
  let html = "";
  const server = http.createServer((req, res) => {
    const url = req.url.split("?")[0];
    if (url === "/" || url === "/page") {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end(html);
      return;
    }
    const m = url.match(/^\/(ai|real|hardreal)\/(.+)$/);
    if (!m) { res.writeHead(404); res.end(); return; }
    const file = path.join(baseDir(m[1]), m[2]);
    if (!fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { "Content-Type": "image/jpeg", "Access-Control-Allow-Origin": "*" });
    fs.createReadStream(file).pipe(res);
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const port = server.address().port;

  html = `<!doctype html><html><body>
    ${pairs.map(([c, f]) => `<img id="${c}_${f}" src="http://127.0.0.1:${port}/${c}/${f}" width="320">`).join("\n")}
  </body></html>`;
  fs.writeFileSync(path.join(os.tmpdir(), "ias_e2e.html"), html);

  const puppeteer = require("puppeteer");
  const browser = await puppeteer.launch({
    headless: "new",
    args: [
      `--disable-extensions-except=${EXT}`,
      `--load-extension=${EXT}`,
      "--no-sandbox",
    ],
  });
  const page = await browser.newPage();
  if (EP) {
    // set the forced-EP override from the service-worker context via the
    // extension's own storage API (offscreen reads it at init)
    const swTarget = await browser.waitForTarget(
      (t) => t.type() === "service_worker" && t.url().includes("background"),
      { timeout: 15000 },
    );
    const sw = await swTarget.worker();
    await sw.evaluate(async (ep) => {
      await chrome.storage.local.set({ ias_force_ep: ep });
    }, EP);
    console.log(`forced EP: ${EP}`);
  }
  await page.goto(`http://127.0.0.1:${port}/page`, { waitUntil: "networkidle0" });

  // wait for badges (tolerant poll — headless CfT waitForFunction is flaky)
  for (let i = 0; i < 24; i++) {
    const n = await page.evaluate(() => document.querySelectorAll("[data-ias]").length);
    if (n >= 2 * N) break;
    await new Promise((r) => setTimeout(r, 5000));
  }
  await new Promise((r) => setTimeout(r, 3000)); // let inference settle

  // runtime measurement (SW is alive now — it just relayed the inferences)
  let perf = [];
  try {
    const t = browser.targets().find((t) => t.type() === "service_worker" && t.url().includes("background"));
    if (t) {
      perf = await (await t.worker()).evaluate(() =>
        new Promise((resolve) => chrome.runtime.sendMessage({ type: "ias:perf" }, (r) => resolve((r && r.samples) || []))),
      );
    }
  } catch (e) { console.log("perf unavailable: " + (e?.message || e)); }

  // match badges to reference predictions by image src (completion order != DOM order)
  const badges = await page.evaluate(() =>
    [...document.querySelectorAll("[data-ias]")].map((el) => ({
      src: el.getAttribute("data-ias-src"),
      score: Number(el.getAttribute("data-ias")) / 100,
      text: el.textContent,
    })),
  );
  const srcToIndex = new Map(pairs.map(([, f], i) => [f, i]));
  const byIndex = new Array(2 * N).fill(null);
  for (const b of badges) {
    const f = (b.src || "").split("/").pop();
    const idx = srcToIndex.get(f);
    if (idx != null) byIndex[idx] = b;
  }
  console.log("extension badges:", byIndex.map((b) => b?.text || "MISSING").join(", "));
  console.log("RAW badges:", JSON.stringify(badges.map(b => [b.src.split("/").pop(), b.score])));
  console.log("refs:", JSON.stringify(refs));
  let maxDelta = 0;
  let meanDelta = 0;
  let clsAgree = 0;
  byIndex.forEach((b, i) => {
    if (!b) return;
    const delta = Math.abs(b.score - refs[i]);
    maxDelta = Math.max(maxDelta, delta);
    meanDelta += delta;
    // classification parity at the 0.65 confidence threshold is the contract
    const extCls = b.score >= 0.65;
    const refCls = refs[i] >= 0.65;
    if (extCls === refCls) clsAgree++;
  });
  meanDelta /= pairs.length;
  console.log(`max |extension - harness| = ${maxDelta.toFixed(3)}  mean = ${meanDelta.toFixed(3)}  cls-agree = ${clsAgree}/${pairs.length}`);
  if (perf.length) {
    perf.sort((a, b) => a - b);
    const p50 = perf[Math.floor(perf.length / 2)];
    const p95 = perf[Math.min(perf.length - 1, Math.floor(perf.length * 0.95))];
    console.log(`perf ms: n=${perf.length} p50=${p50.toFixed(1)} p95=${p95.toFixed(1)} max=${perf[perf.length - 1].toFixed(1)}`);
  } else {
    console.log("perf unavailable (no samples)");
  }
  // quantized/fp16 inference is allowed a confidence band; the decision must match
  const pass = byIndex.every(Boolean) && clsAgree >= 0.98 * pairs.length && meanDelta < 0.05;
  console.log(pass ? "E2E PASS" : "E2E FAIL");
  await browser.close();
  server.close();
  process.exit(pass ? 0 : 1);
}

main().catch((e) => { console.error(e); process.exit(1); });
