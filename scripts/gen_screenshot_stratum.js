#!/usr/bin/env node
/* Generate a screenshot/UI-embedded stratum: render eval images inside
 * realistic web pages and screenshot them (re-photography via the browser
 * pipeline: CSS scaling, JPEG re-encode, text/UI chrome around the image).
 *
 * Evaluation-only diagnostic — NEVER training data. Measures how the
 * detector handles images that have been through a screen-capture path.
 *
 * Usage: node scripts/gen_screenshot_stratum.js [--n 50] [--out data/probe2/screenshots]
 */
const path = require("node:path");
const fs = require("node:fs");
const os = require("node:os");
const http = require("node:http");

const ROOT = path.resolve(__dirname, "..");
const N = parseInt(process.argv.find((a) => a.startsWith("--n="))?.split("=")[1] || "50", 10);
const OUT_ARG = process.argv.find((a) => a.startsWith("--out="))?.split("=")[1] || "data/probe2/screenshots";
const OUT = OUT_ARG.startsWith("/") ? OUT_ARG : path.resolve(ROOT, OUT_ARG);
// --src: training data root (defaults to the sealed eval); --train uses data/train
const SRC = process.argv.find((a) => a.startsWith("--src="))?.split("=")[1] || "data/eval";
const EVAL_DIR = SRC.startsWith("/") ? SRC : path.join(ROOT, SRC);

const aiFiles = fs.readdirSync(path.join(EVAL_DIR, "ai")).slice(0, N);
const realFiles = fs.readdirSync(path.join(EVAL_DIR, "real")).slice(0, N);
const pairs = [
  ...aiFiles.map((f) => ["ai", f]),
  ...realFiles.map((f) => ["real", f]),
];

const server = http.createServer((req, res) => {
  const m = req.url.match(/^\/(ai|real)\/(.+)$/);
  if (!m) { res.writeHead(404); res.end(); return; }
  const file = path.join(EVAL_DIR, m[1], m[2]);
  if (!fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { "Content-Type": "image/jpeg" });
  fs.createReadStream(file).pipe(res);
});

function pageHtml(name) {
  return `<!doctype html><html><head><style>
    body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #f2f2f4; }
    .card { background:#fff; margin: 18px auto; max-width: 700px; padding: 14px;
            border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.12); }
    .card img { width: 100%; border-radius: 6px; display: block; }
    .meta { color: #444; font-size: 14px; margin-top: 10px; }
    .bar { height: 34px; background: #f8f9fa; border-bottom: 1px solid #ddd; display:flex; align-items:center; padding:0 16px; font-size:13px; color:#333; }
  </style></head><body>
  <div class="bar">Example News — today's story</div>
  <div class="card">
    <img src="http://127.0.0.1:${PORT}/${name}" width="700">
    <div class="meta">Posted by reader · 12 comments · share</div>
  </div>
  </body></html>`;
}

async function main() {
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  global.PORT = server.address().port;
  fs.mkdirSync(OUT, { recursive: true });

  const puppeteer = require("puppeteer");
  // restart the browser every 25 shots: long setContent loops leak memory and
  // Chromium dies mid-run on large source images
  let browser = null;
  let page = null;
  let sinceRestart = 25;
  const ensureBrowser = async () => {
    if (sinceRestart < 25 && browser) return;
    if (browser) { await browser.close().catch(() => {}); }
    browser = await puppeteer.launch({ headless: "new", args: ["--no-sandbox", "--disable-dev-shm-usage"] });
    page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 1 });
    sinceRestart = 0;
  };

  let done = 0;
  for (const [cls, f] of pairs) {
    await ensureBrowser();
    const html = pageHtml(`${cls}/${f}`);  // pageHtml closes over PORT via global
    await page.setContent(html, { waitUntil: "networkidle0" });
    // scroll a touch for realism, then capture the card region
    await page.evaluate(() => window.scrollTo(0, 0));
    await new Promise((r) => setTimeout(r, 120));
    const outFile = path.join(OUT, `${cls}_shot_${f.replace(/\.[^.]+$/, "")}.png`);
    await page.screenshot({ path: outFile, type: "png" });
    done++;
    if (done % 25 === 0) console.log(`${done}/${pairs.length} screenshots`);
  }
  await browser.close();
  server.close();
  console.log(`wrote ${done} screenshots to ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
