#!/usr/bin/env python3
"""Generate the authoritative model-bundle manifest.

Reads every runtime model artifact in detector/, computes SHA-256 hashes and
ORT input/output shapes, and writes:
  - detector/model_bundle_manifest.json  (authoritative contract)
  - detector/model_cnn.json              (legacy metadata, now correct)

Python (detector/detect.py) and JavaScript (extension/offscreen.js) must
read their preprocessing constants from this manifest — never hard-code
them independently.

Usage:
  python3 scripts/gen_model_manifest.py [--strict]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DET = ROOT / "detector"

# models that must agree on the input shape contract
STUDENT_FILES = ["model_cnn.onnx", "model_cnn_fp16.onnx", "model_cnn_q.onnx"]
TEACHER_FILES = ["model_cnn_teacher.onnx", "model_cnn_teacher_fp16.onnx", "model_cnn_teacher_q.onnx"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ortv_shape(p: Path) -> list[int]:
    import onnx

    try:
        m = onnx.load(str(p))
        dims = m.graph.input[0].type.tensor_type.shape.dim
        return [int(d.dim_value) if d.HasField("dim_value") else str(d.dim_param) for d in dims]
    except Exception as e:
        return {"error": str(e)[:200]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit nonzero on any inconsistency")
    args = ap.parse_args()

    # preprocessing contract — source of truth (matches training val transform)
    contract = {
        "version": 2,
        "input_size": 384,
        "resize_to": 416,  # resize shortest side, then center-crop input_size
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "output_semantics": "P(AI) in [0,1]; 2-class softmax over logits",
        "threshold": 0.65,
        "fusion": "max(P_AI(student), P_AI(teacher)) when teacher present",
    }

    models = {}
    problems = []
    for role, names in (("student", STUDENT_FILES), ("teacher", TEACHER_FILES)):
        for name in names:
            p = DET / name
            if not p.exists():
                problems.append(f"missing {name}")
                continue
            shape = ortv_shape(p)
            models[name] = {
                "role": role,
                "size_bytes": p.stat().st_size,
                "sha256": sha256(p),
                "input_shape": shape,
            }
            if isinstance(shape, list):
                # expect [N, 3, H, W] with H=W=input_size
                if len(shape) == 4 and shape[2:] != [contract["input_size"], contract["input_size"]]:
                    problems.append(f"{name}: input HxW {shape[2:]} != {contract['input_size']}")
                if len(shape) != 4:
                    problems.append(f"{name}: unexpected input rank {len(shape)}")

    manifest = {"contract": contract, "models": models, "generated_by": "scripts/gen_model_manifest.py"}
    out = DET / "model_bundle_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")

    # legacy model_cnn.json — keep in sync (backbone + sizes)
    legacy = {
        "size": contract["input_size"],
        "resize_to": contract["resize_to"],
        "mean": contract["mean"],
        "std": contract["std"],
        "version": 2,
        "backbone": "resnet18 (student) + convnext_tiny (teacher, ensemble)",
        "classes": 2,
        "fusion": contract["fusion"],
    }
    (DET / "model_cnn.json").write_text(json.dumps(legacy, indent=2) + "\n")
    print(f"wrote {DET / 'model_cnn.json'}")

    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        if args.strict:
            sys.exit(1)
    else:
        print("bundle contract OK")


if __name__ == "__main__":
    main()
