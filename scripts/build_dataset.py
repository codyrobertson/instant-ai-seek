#!/usr/bin/env python3
"""Assemble the deterministic benchmark splits from raw sources.

Sources:
  - AI class:   data/raw/ai/*.jpg  (fal flux/schnell, 1024px)
  - Real class: imagenette2-320 (download via scripts/fetch_real.sh)

Outputs (all resized to max-side 512, JPEG q90, committed):
  data/eval/real/  data/eval/ai/   data/train/real/  data/train/ai/
  data/manifests/dataset.csv   (split,label,path)

Splits (fixed seed):
  - real eval: 150 sampled from imagenette VAL (unseen by training)
  - real train: 150 sampled from imagenette TRAIN
  - ai: 300 generated, seeded shuffle -> 150 train / 150 eval
"""
import csv
import random
from pathlib import Path

from PIL import Image

SEED = 20260813
EVAL_PER_CLASS = 150
TRAIN_PER_CLASS = 150
MAX_SIDE = 512
ROOT = Path(__file__).resolve().parent.parent
IMAGENETTE = Path("/tmp/f01d-datasets/imagenette2-320")


def load_imagenette_paths(split: str) -> list[Path]:
    base = IMAGENETTE / split
    return sorted(p for p in base.glob("*/*") if p.suffix.lower() in (".jpg", ".jpeg"))


def resize_save(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        im.save(dst, "JPEG", quality=90)


def main() -> None:
    rng = random.Random(SEED)

    # --- real class from imagenette (val -> eval, train -> train) ---
    real_eval_src = load_imagenette_paths("val")
    real_train_src = load_imagenette_paths("train")
    assert len(real_eval_src) >= EVAL_PER_CLASS and len(real_train_src) >= TRAIN_PER_CLASS, (
        f"imagenette too small: val={len(real_eval_src)} train={len(real_train_src)}"
    )
    real_eval = rng.sample(real_eval_src, EVAL_PER_CLASS)
    real_train = rng.sample(real_train_src, TRAIN_PER_CLASS)

    # --- ai class from generated pool ---
    ai_pool = sorted(Path(ROOT / "data" / "raw" / "ai").glob("*.jpg"))
    assert len(ai_pool) >= EVAL_PER_CLASS + TRAIN_PER_CLASS, (
        f"only {len(ai_pool)} AI images; run scripts/gen_ai_images.py first"
    )
    shuffled = ai_pool[:]
    rng.shuffle(shuffled)
    ai_train, ai_eval = shuffled[:TRAIN_PER_CLASS], shuffled[TRAIN_PER_CLASS:TRAIN_PER_CLASS + EVAL_PER_CLASS]

    rows = []
    for label, samples, split in [("real", real_eval, "eval"), ("real", real_train, "train"),
                                  ("ai", ai_eval, "eval"), ("ai", ai_train, "train")]:
        for i, src in enumerate(samples, 1):
            dst = ROOT / "data" / split / label / f"{label}_{i:04d}.jpg"
            resize_save(src, dst)
            rows.append({"split": split, "label": label, "path": dst.relative_to(ROOT).as_posix()})

    manifest = ROOT / "data" / "manifests" / "dataset.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "label", "path"])
        w.writeheader()
        w.writerows(rows)

    counts = {s: {l: 0 for l in ("real", "ai")} for s in ("eval", "train")}
    for r in rows:
        counts[r["split"]][r["label"]] += 1
    print(f"wrote {manifest}")
    print(f"eval : real={counts['eval']['real']} ai={counts['eval']['ai']}")
    print(f"train: real={counts['train']['real']} ai={counts['train']['ai']}")


if __name__ == "__main__":
    main()
