// ias content script: scans page images, requests inference, overlays badges
// and blurs AI images (hover reveals). Privacy: image pixels are sent only to
// the extension's own offscreen document; nothing leaves the browser.

(() => {
  if (window.__iasInstalled) return;
  window.__iasInstalled = true;

  const MIN_SIDE = 64;          // ignore icons/spinners/blank trackers
  const MAX_SEND_SIDE = 768;    // downscale before messaging
  const BLUR_PX = 12;
  const BADGE_STYLE = `
    position: absolute !important;
    z-index: 2147483647 !important;
    font: 600 11px/1.4 -apple-system, "Segoe UI", Roboto, sans-serif !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    color: #fff !important;
    background: rgba(0,0,0,.75) !important;
    pointer-events: none !important;
    letter-spacing: .2px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.4) !important;
  `;

  const scored = new Set(); // keys: img src

  const debug = (extra) => {
    document.body.dataset.iasDebug = JSON.stringify({
      scanned: scored.size,
      images: document.images.length,
      ts: Date.now(),
      ...extra,
    });
  };

  function keyOf(img) {
    return img.currentSrc || img.src || "";
  }

  function fitsCriteria(img) {
    if (!img.isConnected) return false;
    const w = img.naturalWidth || img.width || 0;
    const h = img.naturalHeight || img.height || 0;
    if (Math.min(w, h) < MIN_SIDE) return false;
    const k = keyOf(img);
    if (!k || scored.has(k)) return false;
    if (k.startsWith("data:") && k.length < 800) return false;
    return true;
  }

  function toJpegDataUrl(img, maxSide) {
    const w = img.naturalWidth, h = img.naturalHeight;
    const scale = Math.min(1, maxSide / Math.max(w, h));
    const cw = Math.max(1, Math.round(w * scale));
    const ch = Math.max(1, Math.round(h * scale));
    const canvas = document.createElement("canvas");
    canvas.width = cw; canvas.height = ch;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, cw, ch);
    try {
      return canvas.toDataURL("image/jpeg", 0.9);
    } catch {
      return null; // tainted canvas: can't read pixels
    }
  }

  function badgeFor(img, pAi, rawSide) {
    const state = pAi >= 0.65 ? "ai" : pAi < 0.35 ? "real" : "uncertain";
    const label = state === "ai" ? "AI" : state === "real" ? "real" : "?";
    const el = document.createElement("div");
    el.textContent = `${label} ${Math.round(pAi * 100)}%`;
    el.style.cssText = BADGE_STYLE +
      (state === "ai" ? "background: rgba(180,32,32,.88) !important;"
       : state === "real" ? "background: rgba(24,110,60,.85) !important;"
       : "background: rgba(140,110,20,.88) !important;");
    const rect = img.getBoundingClientRect();
    el.style.top = Math.max(0, rect.top + window.scrollY) + "px";
    el.style.left = Math.max(0, rect.left + window.scrollX) + "px";
    el.style.maxWidth = Math.max(60, rect.width) + "px";
    el.setAttribute("data-ias", String(Math.round(pAi * 100)));
    el.setAttribute("data-ias-side", String(rawSide));
    document.body.appendChild(el);

    // blur AI images (hover reveals the original).
    // No transition: a filter animation re-rasterizes every frame; a single
    // filter assignment composites once on the GPU (faster than inference).
    // Uncertain (0.35..0.65) images are NOT blurred — the product abstains.
    if (state === "ai" && !img.dataset.iasBlurred) {
      img.dataset.iasBlurred = "1";
      img.style.willChange = "filter";
      const applyBlur = () => { img.style.filter = `blur(${BLUR_PX}px)`; };
      const clearBlur = () => { img.style.filter = ""; };
      applyBlur();
      img.addEventListener("mouseenter", clearBlur, { passive: true });
      img.addEventListener("mouseleave", applyBlur, { passive: true });
      img.style.cursor = "pointer";
    }
    return el;
  }

  function repositionBadges() {
    document.querySelectorAll("[data-ias]").forEach((el) => {
      const k = el.getAttribute("data-ias-src");
      if (!k) return;
      const img = [...document.images].find((i) => keyOf(i) === k);
      if (!img || !img.isConnected) { el.remove(); return; }
      const rect = img.getBoundingClientRect();
      el.style.top = Math.max(0, rect.top + window.scrollY) + "px";
      el.style.left = Math.max(0, rect.left + window.scrollX) + "px";
    });
  }

  async function scoreImage(img) {
    const k = keyOf(img);
    scored.add(k);
    const dataUrl = toJpegDataUrl(img, MAX_SEND_SIDE);
    if (!dataUrl) return;
    const side = Math.max(img.naturalWidth, img.naturalHeight);
    const rawSide = Math.round(Math.min(side, MAX_SEND_SIDE));
    const resp = await new Promise((resolve) => {
      chrome.runtime.sendMessage({ type: "ias:infer", dataUrl, side: rawSide }, (r) => {
        resolve(r ?? { error: chrome.runtime.lastError?.message || "no response" });
      });
    });
    debug({ lastError: resp?.error || null, lastPai: typeof resp?.pAi === "number" ? Math.round(resp.pAi * 100) : null });
    if (resp && typeof resp.pAi === "number") {
      const el = badgeFor(img, resp.pAi, rawSide);
      el.setAttribute("data-ias-src", k);
    }
  }

  function scan() {
    try {
      for (const img of document.images) {
        if (fitsCriteria(img)) {
          scoreImage(img).catch(() => {});
        }
      }
      debug({});
    } catch (e) {
      debug({ error: String(e.message || e) });
    }
  }

  // re-flow badges on scroll/resize (throttled)
  let repositioning = false;
  const onViewChange = () => {
    if (repositioning) return;
    repositioning = true;
    requestAnimationFrame(() => { repositionBadges(); repositioning = false; });
  };
  window.addEventListener("scroll", onViewChange, { passive: true });
  window.addEventListener("resize", onViewChange, { passive: true });

  const observer = new MutationObserver(() => scan());
  observer.observe(document.documentElement, { childList: true, subtree: true });

  // images loaded after initial scan (lazy loading)
  document.addEventListener("load", (e) => {
    if (e.target instanceof HTMLImageElement) scan();
  }, true);

  scan();
})();
