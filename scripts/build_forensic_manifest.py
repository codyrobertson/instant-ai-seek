#!/usr/bin/env python3
"""Build provenance-safe forensic manifests from the training data tree.

Walks the Spark data layout (repo root or --root), assigns strata (family,
source, license, prompt_id when known), hashes every image, and emits:
  data/manifests/train_forensic.csv
  data/manifests/dev_forensic.csv
  data/manifests/sealed_forensic.csv
  data/manifests/forensic_registry.json

Split policy (must match detector/train_cnn.py assemble_rows):
  - dev = 20% stratified per source (deterministic, SEED=0)
  - sealed = holdout_oos + never-trainable sets (bench parquets are NOT
    listed here — they are law-eval only; see benchmark_registry.json)

Usage:
  python3 scripts/build_forensic_manifest.py --root <data-root> [--no-hash]
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = Path(__file__).resolve().parent.parent / "data/manifests/forensic_registry.schema.json"
SEED = 0

# source -> (family, license)
SOURCE_META = {
    # real
    "imagenette": ("web_jpeg", "Apache-2.0 (ImageNet-derived)"),
    "picsum": ("camera_photo", "Unsplash license"),
    "ffhq": ("portrait", "FFHQ license (research)"),
    "cifake_real": ("camera_photo", "MIT (CIFAKE)"),
    "probe_real": ("camera_photo", "web scraped, research"),
    "screenshots": ("screenshot_ui", "generated in-repo"),
    # ai
    "schnell": ("latent_diffusion", "fal.ai TOS (research)"),
    "xgen": ("latent_diffusion", "fal.ai TOS (research)"),
    "probe_ai": ("latent_diffusion", "fal.ai TOS (research)"),
    "mj": ("commercial_api", "web scraped (research, no redistribution)"),
    "mj2": ("commercial_api", "web scraped (research, no redistribution)"),
    "sd_db": ("latent_diffusion", "DiffusionDB research license"),
    "stylegan3": ("gan", "NVlabs StyleGAN3 license (research)"),
    "cf_fakes": ("gan", "Community Forensics research license"),
    "cifake_ai": ("latent_diffusion", "MIT (CIFAKE)"),
    "real_coco": ("camera_photo", "COCO (CC-BY-4.0) — not used in training (reverted run)"),
}

# filesystem -> source mapping for the Spark tree
DIR_TO_SOURCE = {
    "imagenette2-320": "imagenette",
    "picsum": "picsum",
    "ffhq": "ffhq",
    "cifake": "cifake_real",       # cifake/<real>/... assumed below
    "probe": "probe_real",
    "screenshots": "screenshots",
    "schnell": "schnell",
    "xgen": "xgen",
    "probe_ai": "probe_ai",
    "mj": "mj",
    "mj2": "mj2",
    "sd_db": "sd_db",
    "stylegan3": "stylegan3",
    "cf_fakes": "cf_fakes",
    "holdout_oos": "mj2|sd_db",     # sealed, special-cased
    # Spark train_extra names
    "ai_cf": "cf_fakes",
    "ai_cf1": "cf_fakes",
    "ai_cf2": "cf_fakes",
    "ai_cf3": "cf_fakes",
    "ai_cf4": "cf_fakes",
    "ai_cf5": "cf_fakes",
    "ai_cf6": "cf_fakes",
    "ai_mj": "mj",
    "ai_mj2_train": "mj2",
    "ai_sd_db": "sd_db",
    "ai_stylegan3": "stylegan3",
    "ai_cifake": "cifake_ai",
    "real_cifake": "cifake_real",
    "real_ffhq": "ffhq",
    "real_ffhq_dl": "ffhq",
    "real_ffhq_dl2": "ffhq",
    "real_ffhq_dl3": "ffhq",
    "real_coco": "real_coco",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(path: Path, roots: list[Path]) -> dict:
    """Return {source, family, license, label} for a path under one of the roots."""
    rel = path
    for root in roots:
        try:
            rel = path.relative_to(root)
            break
        except ValueError:
            continue
    parts = rel.parts
    # sealed holdout_oos: dirs ai_mj2 / ai_sd_db
    if parts[0] == "holdout_oos" and len(parts) > 1:
        sub = parts[1]
        if "mj" in sub:
            return {"source": "mj2", "family": "commercial_api", "license": SOURCE_META["mj2"][1], "label": "ai"}
        return {"source": "sd_db", "family": "latent_diffusion", "license": SOURCE_META["sd_db"][1], "label": "ai"}
    # top-level dirs
    top = parts[0]
    if top == "train_extra" and len(parts) > 1:
        # train_extra/<source_dir>/<file>
        inner = parts[1]
        source = DIR_TO_SOURCE.get(inner, inner)
        family, license_ = SOURCE_META.get(source, ("unknown", "unknown"))
        label = "ai" if family in ("latent_diffusion", "pixel_diffusion", "gan", "autoregressive", "commercial_api", "img2img", "postprocessed") else "real"
        if inner in ("real_coco",) :
            family, license_ = "camera_photo", SOURCE_META["real_coco"][1] if "real_coco" in SOURCE_META else "unknown"
        return {"source": source, "family": family, "license": license_, "label": label}
    if top == "xgen" and len(parts) > 2:
        # xgen/train/<model> — training-time generations
        return {"source": "xgen", "family": "latent_diffusion",
                "license": SOURCE_META["xgen"][1], "label": "ai"}
    if top == "train":
        # train/{ai,real} — snapshot samples used in training
        label = "ai" if len(parts) > 1 and parts[1] == "ai" else "real"
        return {"source": "schnell" if label == "ai" else "imagenette",
                "family": SOURCE_META["schnell" if label == "ai" else "imagenette"][0],
                "license": SOURCE_META["schnell" if label == "ai" else "imagenette"][1],
                "label": label}
    if top in DIR_TO_SOURCE:
        source = DIR_TO_SOURCE[top]
    else:
        source = top
    family, license_ = SOURCE_META.get(source, ("unknown", "unknown"))
    label = "ai" if family in ("latent_diffusion", "pixel_diffusion", "gan", "autoregressive", "commercial_api", "img2img", "postprocessed") else "real"
    if source == "cifake_real" and len(parts) > 1 and parts[1] == "ai":
        source = "cifake_ai"
        family, license_ = SOURCE_META["cifake_ai"]
        label = "ai"
    if source == "probe_real" and len(parts) > 1 and parts[1] == "ai":
        source = "probe_ai"
        family, license_ = SOURCE_META["probe_ai"]
        label = "ai"
    return {"source": source, "family": family, "license": license_, "label": label}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, action="append",
                    help="training data root (DGX Spark layout); repeatable")
    ap.add_argument("--no-hash", action="store_true", help="skip SHA-256 (fast pass)")
    ap.add_argument("--max-rows", type=int, default=0, help="cap rows per source (0 = all)")
    args = ap.parse_args()

    roots = [Path(r).expanduser() for r in args.root]
    for root in roots:
        if not root.exists():
            sys.exit(f"data root not found: {root}")

    exts = (".jpg", ".jpeg", ".png", ".webp")
    # NEVER sweep sealed/bench suites into the training manifest
    EXCLUDE_DIRS = {"eval", "holdout", "holdout2", "bench_pub", "probe", "raw"}
    files = []
    for root in roots:
        for p in root.rglob("*"):
            if not (p.is_file() and p.suffix.lower() in exts):
                continue
            rel = p.relative_to(root)
            if rel.parts and rel.parts[0] in EXCLUDE_DIRS:
                continue
            files.append(p)
    print(f"found {len(files)} images under {roots} (excluded {EXCLUDE_DIRS})")

    rng = random.Random(SEED)
    rows = []
    by_source = {}
    for p in files:
        meta = classify(p, roots)
        key = (meta["source"], meta["label"])
        by_source.setdefault(key, []).append(p)
    for key, ps in by_source.items():
        rng.shuffle(ps)
        if args.max_rows:
            ps = ps[: args.max_rows]
        for p in ps:
            meta = classify(p, roots)
            row = {
                "path": str(p),
                "label": meta["label"],
                "family": meta["family"],
                "source": meta["source"],
                "license": meta["license"],
                "split": "",  # filled below
                "prompt_id": None,
                "scene_group": None,
                "transform_tags": "[]",
                "sha256": "" if args.no_hash else sha256_file(p),
                "size_bytes": p.stat().st_size,
            }
            rows.append(row)

    # dedupe by SHA-256 FIRST (keep first occurrence; dataset-download artifacts)
    seen_h = set()
    deduped = []
    dropped = 0
    for r in rows:
        if r["sha256"] and r["sha256"] in seen_h:
            dropped += 1
            continue
        if r["sha256"]:
            seen_h.add(r["sha256"])
        deduped.append(r)
    if dropped:
        print(f"deduped {dropped} exact-duplicate files")
    rows = deduped

    # sealed: holdout_oos only (never trainable) — excluded from train/dev
    sealed_rows = [r for r in rows if "/holdout_oos/" in r["path"]]
    for r in sealed_rows:
        r["split"] = "sealed"
    pool = [r for r in rows if "/holdout_oos/" not in r["path"]]

    # deterministic stratified dev split (20% per source+label, SEED=0)
    dev_rows, train_rows = [], []
    by_key = {}
    for r in pool:
        by_key.setdefault((r["source"], r["label"]), []).append(r)
    for key, rs in by_key.items():
        rs = sorted(rs, key=lambda r: r["path"])
        n_dev = max(1, round(len(rs) * 0.2))
        rng.shuffle(rs)
        for r in rs[:n_dev]:
            r["split"] = "dev"
            dev_rows.append(r)
        for r in rs[n_dev:]:
            r["split"] = "train"
            train_rows.append(r)

    def write_csv(path: Path, rows_: list[dict]) -> None:
        if not rows_:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0].keys()))
            w.writeheader()
            for r in rows_:
                w.writerow(r)

    out = Path(__file__).resolve().parent.parent / "data/manifests"
    write_csv(out / "train_forensic.csv", train_rows)
    write_csv(out / "dev_forensic.csv", dev_rows)
    write_csv(out / "sealed_forensic.csv", sealed_rows)

    # registry json (schema-compliant)
    reg = {
        "schema_version": 1,
        "generated_by": "scripts/build_forensic_manifest.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "splits": {
            "train": {"manifest": "train_forensic.csv", "rows": len(train_rows)},
            "dev": {"manifest": "dev_forensic.csv", "rows": len(dev_rows)},
            "sealed": {"manifest": "sealed_forensic.csv", "rows": len(sealed_rows)},
        },
    }
    (out / "forensic_registry.json").write_text(json.dumps(reg, indent=2))

    from collections import Counter

    print("train rows:", len(train_rows))
    print("dev rows:", len(dev_rows))
    print("sealed rows:", len(sealed_rows))
    print("by source+label:")
    for k, v in sorted(Counter((r["source"], r["label"]) for r in train_rows + dev_rows).items()):
        print(f"  {k[0]:16s} {k[1]:5s} {v}")


if __name__ == "__main__":
    main()
