#!/usr/bin/env python3
"""Tournament trainer for the hybrid teacher families.

T1 — multi-view:      --views full,hp,lp (semantic + frequency experts)
T2 — artifact-resist: add --patch-shuffle (scene-randomization aug)
T3 — generator-aware: add --prototypes (auxiliary source-prototype head)

Same data/split/budget/calibration policy as train_cnn.py (imports its
assembly). Deterministic per --seed. Best checkpoint by dev val_bal.

Usage:
  python3 detector/train_hybrid.py --views full,hp,lp --epochs 26 --batch 64 \
      --input-size 384 --output runs/t1_seed0 [--patch-shuffle] [--prototypes]
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.model_hybrid import HybridTeacher, hp_view  # noqa: E402
from detector.train_cnn import (  # noqa: E402
    AIDataset,
    MEAN,
    STD,
    assemble_rows,
)
from torch.utils.data import DataLoader  # noqa: E402

# source -> prototype id (stable across runs; built at startup from train rows)
SOURCE_IDS: dict[str, int] = {}


def build_source_ids(train_rows: list[dict]) -> None:
    global SOURCE_IDS
    SOURCE_IDS = {s: i for i, s in enumerate(sorted({r["source"] for r in train_rows}))}


def patch_shuffle(x: torch.Tensor, grid: int = 4, p: float = 0.5) -> torch.Tensor:
    """Scene-randomization: shuffle a grid of patches (destroys semantics)."""
    if random.random() > p:
        return x
    B, C, H, W = x.shape
    ph, pw = H // grid, W // grid
    xg = x[:, :, : ph * grid, : pw * grid]
    patches = xg.reshape(B, C, grid, ph, grid, pw).permute(0, 1, 2, 4, 3, 5)
    patches = patches.reshape(B, C, grid * grid, ph, pw)
    idx = torch.randperm(grid * grid, device=x.device)
    shuffled = patches[:, :, idx].reshape(B, C, grid, grid, ph, pw)
    shuffled = shuffled.permute(0, 1, 2, 4, 3, 5).reshape(B, C, ph * grid, pw * grid)
    return shuffled


class HybridDataset(AIDataset):
    """AIDataset + per-row source id for the prototype head (shuffle-safe)."""

    def __getitem__(self, i: int):
        x, y = super().__getitem__(i)
        src = SOURCE_IDS.get(self.rows[i]["source"], 0)
        return x, y, torch.tensor(src, dtype=torch.long)


class PrototypeHead(nn.Module):
    def __init__(self, embed_dim: int, n_sources: int, tau: float = 0.07):
        super().__init__()
        self.prototypes = nn.Parameter(F.normalize(torch.randn(n_sources, embed_dim), dim=1))
        self.tau = tau

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = z @ self.prototypes.t() / self.tau
        return logits


def validate(model: nn.Module, val_loader: DataLoader, device: str, amp_dtype) -> tuple[float, float]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            ps.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
            ys.append(yb.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = p >= 0.65
    tpr = ((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())
    tnr = ((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum())
    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, len(y) + 1)
    n_pos = y.sum()
    u = ranks[y[order] == 1].sum() - n_pos * (n_pos + 1) / 2
    auroc = u / (n_pos * (len(y) - n_pos)) if n_pos and (len(y) - n_pos) else 0.5
    return (tpr + tnr) / 2, auroc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="full,hp,lp", help="comma-separated: full,hp,lp")
    ap.add_argument("--epochs", type=int, default=26)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--patch-shuffle", action="store_true", help="T2")
    ap.add_argument("--prototypes", action="store_true", help="T3")
    ap.add_argument("--proto-weight", type=float, default=0.1)
    ap.add_argument("--per-epoch", type=int, default=4000)
    ap.add_argument("--output", required=True, help="run dir (best state + onnx)")
    args = ap.parse_args()

    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # inherit train_cnn's INPUT_SIZE global for AIDataset crops
    import detector.train_cnn as tc

    tc.INPUT_SIZE = args.input_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} views={views} T2_patch_shuffle={args.patch_shuffle} T3_prototypes={args.prototypes} seed={args.seed}")

    train_rows, val_rows = assemble_rows()
    build_source_ids(train_rows)
    n_real = sum(1 for r in train_rows if r["label"] == "real")
    n_ai = sum(1 for r in train_rows if r["label"] == "ai")
    print(f"train: real={n_real} ai={n_ai} | sources={len(SOURCE_IDS)}")

    ai_rows = [r for r in train_rows if r["label"] == "ai"]
    real_rows = [r for r in train_rows if r["label"] == "real"]
    per_epoch = min(len(real_rows), len(ai_rows), args.per_epoch)

    model = HybridTeacher(views=views)
    model.to(device)
    if device == "cuda":
        try:
            model = torch.compile(model, backend="inductor", mode="default")
        except Exception as e:  # noqa: BLE001
            print(f"compile unavailable: {e}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.05)
    steps_per_epoch = 2 * per_epoch // args.batch + 1
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    proto_head = None
    if args.prototypes:
        embed_dim = model.head[0].normalized_shape[0]
        proto_head = PrototypeHead(embed_dim, len(SOURCE_IDS)).to(device)
        opt.add_param_group({"params": proto_head.parameters()})
        print(f"prototype head: {len(SOURCE_IDS)} sources, embed {embed_dim}")

    use_amp = device == "cuda" and args.amp != "off"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda") if (use_amp and amp_dtype == torch.float16) else None

    val_ds = AIDataset(val_rows, train=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=12 if device == "cuda" else 0)

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"bal": -1.0, "state": None, "epoch": -1}
    stalled = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        idx = list(range(len(ai_rows)))
        random.shuffle(idx)
        ai_batch = [ai_rows[i] for i in idx[:per_epoch]]
        idx2 = list(range(len(real_rows)))
        random.shuffle(idx2)
        real_batch = [real_rows[i] for i in idx2[:per_epoch]]
        ds = HybridDataset(ai_batch + real_batch, train=True) if args.prototypes else AIDataset(ai_batch + real_batch, train=True)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                            num_workers=12 if device == "cuda" else 0,
                            persistent_workers=True)
        for batch in loader:
            xb, yb = batch[0].to(device), batch[1].to(device)
            if args.patch_shuffle:
                xb = patch_shuffle(xb)
            opt.zero_grad()
            if use_amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(xb)
                    loss = loss_fn(logits.float(), yb.long())
                    if proto_head is not None:
                        z = model.fused_embedding(xb)
                        proto_loss = nn.functional.cross_entropy(proto_head(z), batch[2].to(device))
                        loss = loss + args.proto_weight * proto_loss
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
            else:
                logits = model(xb)
                loss = loss_fn(logits, yb.long())
                loss.backward()
                opt.step()
            sched.step()
        bal, auroc = validate(model, val_loader, device, amp_dtype)
        print(f"epoch {epoch+1}/{args.epochs} loss={loss.item():.4f} val_bal={bal:.4f} val_auroc={auroc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if bal > best["bal"]:
            best["bal"] = bal
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["epoch"] = epoch + 1
            stalled = 0
        elif args.patience > 0:
            stalled += 1
            if stalled >= args.patience:
                print(f"early stop after {epoch + 1} epochs (best val_bal {best['bal']:.4f} @ {best['epoch']})")
                break

    print(f"best val_bal={best['bal']:.4f} @ epoch {best['epoch']}")
    torch.save(best["state"], out_dir / "best_state.pt")

    # ONNX export (unwrapped from compile; strip compile prefixes)
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    sd = {k.removeprefix("_orig_mod."): v for k, v in best["state"].items()}
    m.load_state_dict(sd)
    m.eval().to("cpu")
    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    onnx_path = out_dir / "model_hybrid.onnx"
    torch.onnx.export(m, dummy, onnx_path, input_names=["image"], output_names=["logit"],
                      opset_version=17, dynamo=False,
                      dynamic_axes={"image": {0: "N"}, "logit": {0: "N"}})
    print(f"exported {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")
    # sanity
    import onnxruntime as ort

    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(2, 3, args.input_size, args.input_size)
    o = s.run(None, {"image": x.numpy()})[0]
    print(f"sanity ORT output: {o.shape}")


if __name__ == "__main__":
    main()
