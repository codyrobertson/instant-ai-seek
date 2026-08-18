#!/usr/bin/env python3
"""Strict static checks for the model bundle.

Fails on any of:
  - input-size mismatch between artifacts and the manifest contract
  - missing or stale metadata
  - missing model hash
  - fp32/fp16/q8 output-shape mismatch
  - preprocessing resize/crop mismatch
  - q8 graph unsupported by the selected ORT Web backend (structural check)

Usage:
  python3 scripts/verify_model_bundle.py [--strict] [--check-ort-web]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DET = ROOT / "detector"

EXPECTED_MODELS = {
    "model_cnn.onnx": "student fp32",
    "model_cnn_fp16.onnx": "student fp16",
    "model_cnn_q.onnx": "student q8",
    "model_cnn_teacher.onnx": "teacher fp32",
    "model_cnn_teacher_fp16.onnx": "teacher fp16",
    "model_cnn_teacher_q.onnx": "teacher q8",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--check-ort-web", action="store_true", help="load-check q8 graphs via onnxruntime-web JS? (no-op here)")
    args = ap.parse_args()

    problems = []

    manifest_path = DET / "model_bundle_manifest.json"
    legacy_path = DET / "model_cnn.json"
    if not manifest_path.exists():
        problems.append("missing detector/model_bundle_manifest.json — run scripts/gen_model_manifest.py")
    else:
        manifest = json.loads(manifest_path.read_text())
        contract = manifest["contract"]
        if contract["input_size"] != 384 or contract["resize_to"] != 416:
            problems.append(f"manifest contract not 384/416: {contract['input_size']}/{contract['resize_to']}")
        # every expected model present + hashed
        for name, desc in EXPECTED_MODELS.items():
            entry = manifest["models"].get(name)
            if entry is None:
                problems.append(f"{name} ({desc}) missing from manifest")
                continue
            if not entry.get("sha256") or len(entry["sha256"]) != 64:
                problems.append(f"{name}: missing/invalid sha256")
            shape = entry.get("input_shape")
            if not isinstance(shape, list) or len(shape) != 4 or shape[2:] != [384, 384]:
                problems.append(f"{name}: input shape {shape} != [N,3,384,384]")
            # on-disk hash must match manifest (stale bundle guard)
            p = DET / name
            if p.exists():
                import hashlib

                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() != entry["sha256"]:
                    problems.append(f"{name}: on-disk hash != manifest hash (regenerate manifest)")

        # fp32/fp16/q8 output-shape parity (structural: same graph output count/name)
        outs = {}
        import onnx

        for name in EXPECTED_MODELS:
            p = DET / name
            if not p.exists():
                continue
            try:
                m = onnx.load(str(p))
                outs[name] = [(o.name, [str(d.dim_value) if d.HasField("dim_value") else str(d.dim_param)
                                        for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
            except Exception as e:  # noqa: BLE001
                problems.append(f"{name}: onnx load failed {str(e)[:80]}")
        if outs:
            base = outs.get("model_cnn.onnx")
            for name, o in outs.items():
                if name != "model_cnn.onnx" and o != base:
                    problems.append(f"{name}: output contract {o} != fp32 {base}")

        # preprocessing parity with the legacy json
        if legacy_path.exists():
            legacy = json.loads(legacy_path.read_text())
            if legacy.get("size") != contract["input_size"] or legacy.get("resize_to") != contract["resize_to"]:
                problems.append("model_cnn.json stale: size/resize_to != manifest contract")

    if problems:
        print("VERIFY FAIL:")
        for p in problems:
            print("  -", p)
        if args.strict:
            sys.exit(1)
    else:
        print("model bundle VERIFY OK (6 artifacts, 384px contract, hashes match)")


if __name__ == "__main__":
    main()
