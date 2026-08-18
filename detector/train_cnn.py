#!/usr/bin/env python3
"""Train the production detector: convnext_tiny fine-tuned on real-vs-AI.

Data:
  real  : imagenette2-320 (train+val) + picsum probe photos
  ai    : flux/schnell train split + cross-generator probes + fresh batches

The held-out benchmark eval (data/eval) NEVER enters training or validation.

Pipeline: deterministic split -> class-balanced epochs with heavy
augmentation (incl. JPEG re-encode at random quality — the key robustness
trick for web images) -> best-val checkpoint -> ONNX export (fp32 + uint8
dynamic quant) -> detector/model_cnn.onnx (+ model_cnn.json metadata).

Usage: python3 detector/train_cnn.py [--epochs 12] [--no-export]
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 0
INPUT_SIZE = 256  # overridden by --input-size; models/export/detect must agree
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGENETTE = Path("/tmp/f01d-datasets/imagenette2-320")


# ---------------- data assembly ----------------

def list_imagenette() -> list[Path]:
    # TRAIN split only: the eval set's real images were sampled from the
    # imagenette VAL split — training on val would leak the eval set.
    return sorted(p for p in IMAGENETTE.glob("train/*/*") if p.suffix.lower() in (".jpg", ".jpeg"))


def _rows(paths: list[Path], label: str, source: str) -> list[dict]:
    return [{"label": label, "source": source, "path": str(p)} for p in paths]


def assemble_rows() -> tuple[list[dict], list[dict]]:
    """Returns (train_rows, val_rows); val = 20% of EVERY source (stratified).

    Sources: imagenette, picsum, cifake_real, schnell, each probe generator,
    cifake_ai, each xgen training generator. Stratification keeps per-class
    val signals honest (e.g. SDXL images are never only in train).
    """
    rng = random.Random(SEED)
    groups: list[list[dict]] = []

    # real sources
    groups.append(_rows(list_imagenette(), "real", "imagenette"))
    groups.append(_rows(sorted((ROOT / "data/probe/real").glob("*.jpg")), "real", "picsum"))
    cifake_real = sorted((ROOT / "data/train_extra/real_cifake").glob("*.jpg"))
    rng.shuffle(cifake_real)
    groups.append(_rows(cifake_real[:2500], "real", "cifake_real"))

    # ai sources
    groups.append(_rows(sorted((ROOT / "data/train/ai").glob("*.jpg")), "ai", "schnell"))
    groups.append(_rows([p for p in sorted((ROOT / "data/raw/ai").glob("*.jpg")) if int(p.stem) > 300],
                        "ai", "schnell"))
    for d in sorted((ROOT / "data/probe/ai").glob("*")):
        if d.is_dir():
            groups.append(_rows(sorted(d.glob("*.jpg")), "ai", f"probe_{d.name}"))
    for d in sorted((ROOT / "data/xgen/train").glob("*")):
        if d.is_dir():
            groups.append(_rows(sorted(d.glob("*.jpg")), "ai", f"xgen_{d.name}"))
    for d in sorted((ROOT / "data/xgen2/train").glob("*")):
        if d.is_dir():
            groups.append(_rows(sorted(d.glob("*.jpg")), "ai", f"xgen2_{d.name}"))
    cifake_ai = sorted((ROOT / "data/train_extra/ai_cifake").glob("*.jpg"))
    rng.shuffle(cifake_ai)
    groups.append(_rows(cifake_ai[:2500], "ai", "cifake_ai"))
    # training subsets only — the SEALED out-of-domain holdout lives in
    # data/holdout_oos/ and must NEVER appear here
    for src_name in ("ai_mj", "ai_mj2_train", "ai_sd_db", "ai_cf", "ai_cf1", "ai_cf2", "ai_cf3", "ai_cf4", "ai_cf5", "ai_cf6"):
        src = ROOT / "data" / "train_extra" / src_name
        files = sorted(p for p in src.glob("*") if p.suffix.lower() in (".jpg", ".png")) if src.exists() else []
        if files:
            groups.append(_rows(files, "ai", src_name))
    # fake faces CAPPED at 6000: the 20k:6.9k real:fake face skew gave the
    # model a "smooth face => fake" prior; balance the face domain instead
    sg = sorted((ROOT / "data" / "train_extra" / "ai_stylegan3").glob("*.jpg"))
    if sg:
        groups.append(_rows(sg[:8000], "ai", "ai_stylegan3"))
    # REAL face portraits (FFHQ) — without real faces in training the model
    # flags smooth real portraits as AI
    ffhq = ROOT / "data" / "train_extra" / "real_ffhq"
    files = sorted(p for p in ffhq.glob("*.png")) if ffhq.exists() else []
    if files:
        groups.append(_rows(files, "real", "ffhq"))


    # REAL browser screenshots of training images (rendered via headless
    # Chromium) — the screenshot stratum measured 14% false-AI on
    # re-photographed reals; simulated render augs failed, real shots work
    ss = ROOT / "data" / "train_extra" / "screenshots_real"
    ss_files = sorted(p for p in ss.glob("*.png")) if ss.exists() else []
    if ss_files:
        groups.append(_rows(ss_files, "real", "screenshots"))
    train, val = [], []
    for g in groups:
        rng.shuffle(g)
        n_val = max(1, len(g) // 5)  # 20% per source
        val.extend(g[:n_val])
        train.extend(g[n_val:])
    return train, val


def noise_quant_map(x: torch.Tensor, n_samples: int = 3, sigma: float = 0.06, bits: int = 5) -> torch.Tensor:
    """CoDA-style probe: per-pixel stability under Gaussian noise + color
    quantization. Real photos (smooth gradients) re-quantize stably; AI
    images (color banding/imbalance) flip more. Returns (B,1,H,W) in [0,1]."""
    q = 2 ** bits
    xc = x.clamp(0, 1)
    ref = (xc * (q - 1)).round()
    flips = torch.zeros_like(xc[:, :1])
    for _ in range(n_samples):
        xn = xc + torch.randn_like(xc) * sigma
        qn = (xn.clamp(0, 1) * (q - 1)).round()
        flips += (qn != ref).float().mean(dim=1, keepdim=True)
    return flips / n_samples


def _browser_render(im: Image.Image, rng=None) -> Image.Image:
    """Screenshot-path simulation: CSS-style bilinear down/up-scale + PNG
    re-encode + slight UI contrast. The screenshot stratum measured 14%
    false-AI on re-photographed reals (browser smoothing reads as
    GAN-smoothness) — give reals the same render artifacts."""
    import io as _io

    import PIL.ImageEnhance as _enhance

    rng = rng or random
    w, h = im.size
    s = rng.uniform(0.45, 0.75)
    im2 = im.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
    im3 = im2.resize((w, h), Image.BILINEAR)
    buf = _io.BytesIO()
    im3.save(buf, "PNG")
    im4 = Image.open(buf).convert("RGB")
    im4 = _enhance.Contrast(im4).enhance(rng.uniform(0.92, 1.08))
    return im4


def _jpeg_reencode(im: Image.Image, q: int) -> Image.Image:
    import io as _io

    buf = _io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return Image.open(buf).convert("RGB")


class _AddGaussianNoise:
    """Sensor-noise simulation: real photos have noise, AI images do not."""

    def __init__(self, sigma: float = 0.08):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.sigma * (torch.rand(1).item() * 0.8 + 0.2)
        return (x + noise).clamp(0, 1)


# ---------------- dataset ----------------

class JpegArtifact:
    """Re-encode through PIL at random quality / format to mimic web images."""

    def __call__(self, im: Image.Image) -> Image.Image:
        r = random.random()
        if r < 0.40:
            q = random.randint(50, 100)
            return self._reencode(im, "JPEG", q)
        if r < 0.50:
            return self._reencode(im, "PNG")
        if r < 0.62:
            return self._reencode(im, "WEBP", random.choice([60, 75, 85, 95]))
        if r < 0.75:
            # downscale then upscale (resampling noise / CDN thumbnails)
            s = random.randint(128, 224)
            return im.resize((s, s), Image.BILINEAR).resize(im.size, Image.BILINEAR)
        return im

    @staticmethod
    def _reencode(im: Image.Image, fmt: str, q: int | None = None) -> Image.Image:
        import io

        buf = io.BytesIO()
        kw = {"quality": q} if q is not None else {}
        im.save(buf, fmt, **kw)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class HeavyWebArtifact:
    """JPEG-compress real faces (CF-Eval real pairs are JPEG-compressed then
    stored as PNG; the model learned "JPEG on a face => AI" because training
    fakes are JPEGs while FFHQ reals are PNGs). Grid search: q60-75 is the
    discriminator band; compress reals there so the cue flips."""

    def __call__(self, im: Image.Image) -> Image.Image:
        q = random.choice([55, 60, 65, 70, 75, 80, 85])
        return JpegArtifact._reencode(im, "JPEG", q)


class AIDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool):
        self.rows = rows
        self.is_train = train
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.4, 1.0), ratio=(0.75, 1.333)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.10),
                transforms.ColorJitter(0.15, 0.15, 0.10, 0.05),
                transforms.RandomApply([JpegArtifact()], p=0.8),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.15),
                transforms.ToTensor(),
                transforms.RandomApply([_AddGaussianNoise()], p=0.25),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(INPUT_SIZE + 32),
                transforms.CenterCrop(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[i]
        with Image.open(row["path"]) as im:
            im = im.convert("RGB")
        # real-side: file-level JPEG compression (q55-85, 40%) — storage cue parity
        if self.is_train and row["label"] == "real" and random.random() < 0.40:
            im = HeavyWebArtifact()(im)
        # NOTE: the AI-side JPEG mirror (t17) and the simulated browser-render
        # aug (t19) were both measured losses — the storage cue is two-sided and
        # simulated render ops don't capture the screenshot path. Real
        # screenshots (screenshots_real source) replace both in teacher20.
        x = self.tf(im)
        y = torch.tensor(1.0 if row["label"] == "ai" else 0.0)
        return x, y


# ---------------- training ----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--model", type=str, default="resnet18",
                    help="timm backbone (resnet18/resnet34/efficientnet_b0/convnext_tiny...)")
    ap.add_argument("--teacher-state", type=str, default=None,
                    help="path to a trained teacher state_dict (distillation)")
    ap.add_argument("--teacher-model", type=str, default="convnext_tiny.in12k_ft_in1k")
    ap.add_argument("--distill-lambda", type=float, default=0.7)
    ap.add_argument("--distill-temp", type=float, default=3.0)
    ap.add_argument("--input-size", type=int, default=256,
                    help="train/infer resolution (384 gives more camera-grain signal)")
    ap.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16",
                    help="mixed precision (bf16 is native on Blackwell; fp16 keeps GradScaler)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop patience in epochs on val_bal (0 = run all epochs)")
    ap.add_argument("--compile-mode", choices=("default", "max-autotune"), default="default",
                    help="torch.compile mode; max-autotune pays off at 384 on fixed shapes")
    ap.add_argument("--feat-distill", type=float, default=0.0,
                    help="feature-distillation weight (moment match: per-channel mean/std of stage-4 "
                         "features via a 1x1 projector; 0 = logit-only KD)")
    ap.add_argument("--probe-input", action="store_true",
                    help="CoDA-style noise-quantization stability map as a 4th input channel")
    ap.add_argument("--hard-negatives", type=str, default=None,
                    help="hard-negative registry CSV (label,path,conf,reason); rows are appended to "
                         "training with per-source weight cap (max 2x ordinary contribution)")
    args = ap.parse_args()
    global INPUT_SIZE
    INPUT_SIZE = args.input_size

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    train_rows, val_rows = assemble_rows()
    if args.hard_negatives:
        import csv as _csv

        hn_rows = []
        with open(args.hard_negatives) as _f:
            for _r in _csv.DictReader(_f):
                if not _r.get("path"):
                    continue
                hn_rows.append({"label": _r["label"], "source": "hard:" + (_r.get("reason") or "review"),
                                "path": _r["path"]})
        # per-source cap: hard rows may double a source's ordinary contribution at most
        from collections import Counter

        ordinary = Counter(r["source"] for r in train_rows)
        capped = []
        hn_by_source = {}
        for r in hn_rows:
            hn_by_source.setdefault(r["source"], []).append(r)
        for s, rs in hn_by_source.items():
            limit = 2 * max(1, ordinary.get(s.replace("hard:", "", 1), 0))
            capped.extend(rs[: max(limit, 200)])
        train_rows.extend(capped)
        print(f"hard negatives: {len(hn_rows)} rows -> {len(capped)} appended (per-source capped)")
    n_real = sum(1 for r in train_rows if r["label"] == "real")
    n_ai = sum(1 for r in train_rows if r["label"] == "ai")
    print(f"train: real={n_real} ai={n_ai} | val: real={sum(1 for r in val_rows if r['label']=='real')} "
          f"ai={sum(1 for r in val_rows if r['label']=='ai')}")

    # class-balanced sampler: sample real down to AI count per epoch
    ai_rows = [r for r in train_rows if r["label"] == "ai"]
    real_rows = [r for r in train_rows if r["label"] == "real"]
    per_epoch = min(len(real_rows), len(ai_rows), 4000)  # cap for epoch speed

    n_workers = 12 if device == "cuda" else 0
    loader_kw = dict(num_workers=n_workers, persistent_workers=n_workers > 0,
                     prefetch_factor=4 if n_workers > 0 else None)
    import timm

    model = timm.create_model(args.model, pretrained=True, num_classes=2)
    if args.probe_input:
        # adapt the first conv to accept the 4th (stability map) channel;
        # zero-init the new channel so pretrained behavior is preserved at start
        import detector.train_cnn as _tc  # noqa: F811

        stem_conv = None
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d) and m.in_channels == 3:
                stem_conv = m
                break
        if stem_conv is None:
            raise RuntimeError("no 3-channel conv found to adapt for probe input")
        new_conv = torch.nn.Conv2d(4, stem_conv.out_channels, stem_conv.kernel_size,
                                   stem_conv.stride, stem_conv.padding, bias=stem_conv.bias is not None)
        with torch.no_grad():
            new_conv.weight[:, :3] = stem_conv.weight
            new_conv.weight[:, 3] = 0.0
            if stem_conv.bias is not None:
                new_conv.bias.copy_(stem_conv.bias)
        parent_name, child_name = name.rsplit(".", 1)
        parent = dict(model.named_modules())[parent_name]
        setattr(parent, child_name, new_conv)
        print("stem adapted for 4-channel probe input")
    model.to(device)
    if device == "cuda":
        try:
            model = torch.compile(model, backend="inductor", mode=args.compile_mode)
            print(f"torch.compile enabled (inductor, mode={args.compile_mode})")
        except Exception as e:  # noqa: BLE001
            print(f"torch.compile unavailable: {e}")
    model.train()

    # teacher for distillation (frozen)
    teacher = None
    feat_proj = None
    if args.teacher_state:
        t = timm.create_model(args.teacher_model, pretrained=False, num_classes=2)
        sd = torch.load(args.teacher_state, map_location="cpu", weights_only=True)
        # state dicts saved from torch.compile'd models carry `_orig_mod.` prefixes
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        t.load_state_dict(sd)
        t.to(device).eval()
        for p in t.parameters():
            p.requires_grad_(False)
        teacher = t
        print(f"teacher loaded from {args.teacher_state}")
        if args.feat_distill > 0:
            # align student features to teacher's via a projector; loss = per-channel
            # mean/std (moment) match over the spatial/token dims, so no spatial
            # alignment needed even at different strides. Supports CNN (B,C,H,W)
            # and ViT-style (B,N,C) feature maps.
            with torch.no_grad():
                _t = teacher.forward_features(torch.zeros(1, 3, args.input_size, args.input_size, device=device))
                _s = model.forward_features(torch.zeros(1, 3, args.input_size, args.input_size, device=device))
            t_feat_ch = _t.shape[1] if _t.dim() == 4 else _t.shape[-1]
            s_feat_ch = _s.shape[1] if _s.dim() == 4 else _s.shape[-1]
            feat_proj = (torch.nn.Conv2d(s_feat_ch, t_feat_ch, 1) if _s.dim() == 4
                         else torch.nn.Linear(s_feat_ch, t_feat_ch)).to(device)
            print(f"feature distillation: student {s_feat_ch}ch -> teacher {t_feat_ch}ch "
                  f"({'CNN' if _s.dim() == 4 else 'ViT'} student, weight {args.feat_distill})")

    opt_params = list(model.parameters()) + (list(feat_proj.parameters()) if feat_proj is not None else [])
    opt = torch.optim.AdamW(opt_params, lr=2e-4, weight_decay=0.05)
    steps_per_epoch = 2 * per_epoch // args.batch + 1
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch, eta_min=1e-6)
    loss_fn = torch.nn.CrossEntropyLoss()

    val_ds = AIDataset(val_rows, train=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, **loader_kw)

    best = {"bal": -1.0, "state": None, "epoch": -1}

    def validate() -> tuple[float, float]:
        model.eval()
        ys, ps = [], []
        val_ctx = (torch.autocast("cuda", dtype=amp_dtype) if use_amp else torch.no_grad())
        with torch.no_grad(), val_ctx:
            for xb, yb in val_loader:
                xb = xb.to(device)
                xin = torch.cat([xb, noise_quant_map(xb)], dim=1) if args.probe_input else xb
                logits = model(xin)
                ps.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
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
        auroc = u / (n_pos * (len(y) - n_pos))
        return (tpr + tnr) / 2, auroc

    use_amp = device == "cuda" and args.amp != "off"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    # fp16 needs a loss scaler; bf16 does not
    scaler = torch.amp.GradScaler("cuda") if (use_amp and amp_dtype == torch.float16) else None
    if use_amp:
        print(f"AMP: {args.amp} (scaler={'on' if scaler else 'off'})")

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
        ds = AIDataset(ai_batch + real_batch, train=True)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                            num_workers=n_workers, persistent_workers=n_workers > 0)
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            if use_amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    xin = torch.cat([xb, noise_quant_map(xb)], dim=1) if args.probe_input else xb
                    logits = model(xin)
                    loss = loss_fn(logits, yb.long())
                    if teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher(xb.float()).float()
                        t_probs = torch.softmax(t_logits / args.distill_temp, dim=1)
                        kl = torch.nn.functional.kl_div(
                            torch.log_softmax(logits.float() / args.distill_temp, dim=1),
                            t_probs, reduction="batchmean") * args.distill_temp ** 2
                        loss = (1 - args.distill_lambda) * loss + args.distill_lambda * kl
                        if args.feat_distill > 0:
                            with torch.no_grad():
                                tf = teacher.forward_features(xb.float())
                            sf = feat_proj(model.forward_features(xb.float()))
                            tf32 = tf.float(); sf32 = sf.float()
                            t_dim = (2, 3) if tf32.dim() == 4 else 1
                            s_dim = (2, 3) if sf32.dim() == 4 else 1
                            tm = tf32.mean(dim=t_dim); ts = sf32.mean(dim=s_dim)
                            tstd = tf32.std(dim=t_dim); sstd = sf32.std(dim=s_dim)
                            loss = loss + args.feat_distill * (
                                torch.nn.functional.mse_loss(ts, tm) + torch.nn.functional.mse_loss(sstd, tstd))
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
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(xb)
                    t_probs = torch.softmax(t_logits / args.distill_temp, dim=1)
                    kl = torch.nn.functional.kl_div(
                        torch.log_softmax(logits / args.distill_temp, dim=1),
                        t_probs, reduction="batchmean") * args.distill_temp ** 2
                    loss = (1 - args.distill_lambda) * loss + args.distill_lambda * kl
                loss.backward()
                opt.step()
            sched.step()
        bal, auroc = validate()
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
                print(f"early stop after {epoch + 1} epochs (val_bal plateau at {best['bal']:.4f})")
                break

    print(f"best val_bal={best['bal']:.4f} @ epoch {best['epoch']}")
    # checkpoint the best state before exporting (export must never lose training)
    torch.save(best["state"], ROOT / "detector/best_state.pt")
    if args.no_export:
        return
    model.load_state_dict(best["state"])
    sanity_image = next(r["path"] for r in val_rows if r["label"] == "real")
    export(model, device, sanity_image, args.model)  # INPUT_SIZE global reflects --input-size
def export(model: torch.nn.Module, device: str, sanity_image: str, backbone: str) -> None:
    import json

    import onnxruntime as ort

    # torch.compile wraps the module — ONNX export needs the original
    model = model._orig_mod if hasattr(model, "_orig_mod") else model
    model.eval().to("cpu")
    n_in = 4 if "--probe-input" in __import__("sys").argv else 3
    dummy = torch.randn(1, n_in, INPUT_SIZE, INPUT_SIZE)
    onnx_path = ROOT / "detector/model_cnn.onnx"
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["image"], output_names=["logit"],
        opset_version=17, dynamo=False,
        dynamic_axes={"image": {0: "N"}, "logit": {0: "N"}},
    )
    print(f"exported {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")

    from onnxruntime.quantization import QuantType, quantize_dynamic

    q_path = ROOT / "detector/model_cnn_q.onnx"
    quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QUInt8)
    print(f"quantized -> {q_path} ({q_path.stat().st_size/1e6:.1f} MB)")

    meta = {"size": INPUT_SIZE, "mean": MEAN, "std": STD, "version": 1,
            "backbone": backbone, "classes": 2}
    (ROOT / "detector/model_cnn.json").write_text(json.dumps(meta, indent=1))

    # sanity: torch vs ORT fp32 vs ORT uint8 on one real and one AI image
    s32 = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sq = ort.InferenceSession(str(q_path), providers=["CPUExecutionProvider"])
    tf = transforms.Compose([
        transforms.Resize(INPUT_SIZE + 32), transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    ai_probe = next(str(p) for p in sorted((ROOT / "data/train/ai").glob("*.jpg")))
    for label, path in (("real", sanity_image), ("ai", ai_probe)):
        with Image.open(path) as im:
            x = tf(im.convert("RGB")).unsqueeze(0).numpy()
        with torch.no_grad():
            t_out = float(torch.softmax(model(torch.from_numpy(x)), dim=1)[0, 1].item())
        o32 = float(torch.softmax(torch.from_numpy(s32.run(None, {"image": x})[0]), dim=1)[0, 1].item())
        oq = float(torch.softmax(torch.from_numpy(sq.run(None, {"image": x})[0]), dim=1)[0, 1].item())
        print(f"sanity {label}: torch={t_out:.4f} fp32={o32:.4f} q8={oq:.4f}")


if __name__ == "__main__":
    main()
