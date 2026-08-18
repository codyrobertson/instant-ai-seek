#!/usr/bin/env python3
"""Pre-registered DINOv3-S full fine-tune (advisor-recommended single attempt).

Design (fixed, no tuning during the run):
  - vit_small_patch16_dinov3 @384, CLS + mean-patch-token features, 2-layer head
  - BCE only; no KD/aux losses; backbone LR 1e-5, head LR 1e-4, wd 0.05,
    5% warmup, cosine decay, grad clip 1.0
  - 12 epochs, checkpoint selected ONLY on the 19.7k dev set
  - threshold chosen on dev: max fake recall subject to real recall >= 0.98
  - evaluate ONCE on the sealed sets (harness + LAW cells)

Gates (from the frozen bundle): harness >= 0.98, CF-Eval real aggregate > 0.37,
probe false-AI < 0.0455, LAW macro improves with no fake cell down > 2pp,
sealed OOS >= 0.995. Any miss ends the foundation direction.

Usage (DGX Spark): python3 bench/dino_probe.py --fine-tune
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.train_cnn import AIDataset, assemble_rows  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="vit_small_patch16_dinov3")
    ap.add_argument("--state", default="/tmp/dinov3_small.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--per-epoch", type=int, default=4000)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--out", default="/tmp/dino_finetune")
    args = ap.parse_args()

    import timm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = timm.create_model(args.teacher, pretrained=False, num_classes=0)
    sd = torch.load(args.state, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
    backbone.load_state_dict(sd, strict=False)

    class Net(nn.Module):
        def __init__(self, back):
            super().__init__()
            self.back = back
            self.head = nn.Sequential(nn.LayerNorm(384 * 2), nn.Linear(384 * 2, 512),
                                      nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 2))

        def forward(self, x):
            f = self.back.forward_features(x)
            feat = torch.cat([f[:, 0], f[:, 1:].mean(dim=1)], dim=1)
            return self.head(feat)

    model = Net(backbone).to(device)
    print(f"model: {args.teacher} fine-tune @{args.input_size} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M)")

    import detector.train_cnn as tc

    tc.INPUT_SIZE = args.input_size
    train_rows, val_rows = assemble_rows()
    rng = np.random.RandomState(0)
    ai = [r for r in train_rows if r["label"] == "ai"]
    real = [r for r in train_rows if r["label"] == "real"]
    ai = [ai[i] for i in rng.permutation(len(ai))[: args.per_epoch]]
    real = [real[i] for i in rng.permutation(len(real))[: args.per_epoch]]
    ds = AIDataset(ai + real, train=True)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=12, persistent_workers=True)
    val_ds = AIDataset(val_rows, train=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=12)

    head_params = list(model.head.parameters())
    back_params = list(model.back.parameters())
    opt = torch.optim.AdamW([
        {"params": back_params, "lr": args.backbone_lr, "base_lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr, "base_lr": args.head_lr},
    ], weight_decay=0.05)
    steps_per_epoch = (2 * args.per_epoch) // args.batch
    total_steps = args.epochs * steps_per_epoch
    warm = int(total_steps * 0.05)

    def lr_at(step):
        if step < warm:
            return step / max(1, warm)
        t = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + np.cos(np.pi * t))

    loss_fn = nn.CrossEntropyLoss()
    best = {"bal": -1.0, "state": None, "epoch": -1}

    def dev_metrics():
        model.eval()
        ys, ps = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for xb, yb in val_loader:
                l = model(xb.to(device))
                ps.append(torch.softmax(l.float(), dim=1)[:, 1].cpu().numpy())
                ys.append(yb.numpy())
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        return y, p

    step = 0
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        tot = 0.0
        for xb, yb in loader:
            mult = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * mult
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(xb), yb.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            tot += loss.item()
        y, p = dev_metrics()
        pred = p >= 0.65
        tpr = ((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())
        tnr = ((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum())
        bal = (tpr + tnr) / 2
        print(f"epoch {ep+1}/{args.epochs} loss={tot/(len(loader)):.4f} dev_bal={bal:.4f} "
              f"ai={tpr:.3f} real={tnr:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if bal > best["bal"]:
            best["bal"] = bal
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["epoch"] = ep + 1
    print(f"best dev_bal={best['bal']:.4f} @ epoch {best['epoch']}")

    # dev-selected threshold: max fake recall with real recall >= 0.98
    model.load_state_dict(best["state"])
    y, p = dev_metrics()
    cands = sorted(set(p), reverse=True)  # descending: highest t with real>=0.98 wins
    best_t, best_fake = 0.65, 0.0
    for t in cands:
        if ((p < t) & (y == 0)).mean() >= 0.98:
            fake = ((p >= t) & (y == 1)).mean()
            if fake > best_fake:
                best_t, best_fake = t, fake
            break  # first (highest) valid threshold is the operating point
    print(f"dev-selected threshold: {best_t:.4f} (fake recall {best_fake:.3f} @ real>=0.98)")

    # save model + threshold
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "best_state.pt")
    (out / "threshold.txt").write_text(str(best_t))
    # ONNX export
    model.eval().to("cpu")
    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    torch.onnx.export(model, dummy, out / "model_dino.onnx", input_names=["image"],
                      output_names=["logit"], opset_version=17, dynamo=False,
                      dynamic_axes={"image": {0: "N"}, "logit": {0: "N"}})
    print(f"saved {out} (best_state.pt, threshold.txt, model_dino.onnx "
          f"{round((out/'model_dino.onnx').stat().st_size/1e6,1)}MB)")


if __name__ == "__main__":
    main()
