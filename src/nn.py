"""Bidirectional GRU on 50 Hz EMG features: day-disjoint CV, final fit, test prediction.

Usage:
    python src/nn.py --name gru_v1                 # 4-fold CV (+ OOF preds saved)
    python src/nn.py --name gru_v1 --final         # also fit on all data, predict tests
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import ACTIONS, CONTEXT, FINGERS, ROOT, cv_score, load_test, load_train, make_folds
from framefeat import HOP, frame_features, frame_targets, upsample

OUT = ROOT / "outputs"


# ----------------------------------------------------------------------------- data
def angle_weights(action: str) -> np.ndarray:
    """Per-angle weights that reproduce the competition's Weighted MAE."""
    if action in FINGERS:
        w = np.full(20, 0.5 / 16, np.float32)
        k = FINGERS.index(action)
        w[4 * k : 4 * k + 4] = 0.5 / 4
        return w
    return np.full(20, 1 / 20, np.float32)


def featurize(recs, cache_name: str):
    path = OUT / "cache" / f"{cache_name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        z = np.load(path)
        if all(f"fr_{r.rid}" in z for r in recs):
            return [z[f"fr_{r.rid}"] for r in recs], [z[f"gl_{r.rid}"] for r in recs]
    fr, gl = [], []
    for r in recs:
        a, b = frame_features(r.x)
        fr.append(a)
        gl.append(b)
    np.savez(path, **{f"fr_{r.rid}": a for r, a in zip(recs, fr)}, **{f"gl_{r.rid}": b for r, b in zip(recs, gl)})
    return fr, gl


# ----------------------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, n_in: int, n_glob: int, h: int = 64, layers: int = 2, dropout: float = 0.2,
                 use_global: bool = True):
        super().__init__()
        self.use_global = use_global
        self.front = nn.Sequential(
            nn.Conv1d(n_in, h, 5, padding=2), nn.GELU(),
            nn.Conv1d(h, h, 5, padding=2), nn.GELU(),
        )
        self.glob = nn.Sequential(nn.Linear(n_glob, 16), nn.GELU()) if use_global else None
        gin = h + (16 if use_global else 0)
        self.gru = nn.GRU(gin, h, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.ctx = nn.Sequential(nn.Linear(2 * h, h), nn.GELU())
        self.head = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, 20))
        self.cls = nn.Linear(2 * h, len(ACTIONS))

    def forward(self, x, g):
        # x: (B, T, D)  g: (B, G)
        z = self.front(x.transpose(1, 2)).transpose(1, 2)
        if self.use_global:
            z = torch.cat([z, self.glob(g)[:, None, :].expand(-1, z.shape[1], -1)], -1)
        o, _ = self.gru(z)
        o = self.drop(o)
        pooled = o.mean(1)
        c = self.ctx(pooled)[:, None, :].expand(-1, o.shape[1], -1)
        y = self.head(torch.cat([o, c], -1))
        return y, self.cls(pooled)


class ResBlock(nn.Module):
    def __init__(self, h, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(h, h, 5, padding=2 * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(h, h, 1)
        self.norm = nn.GroupNorm(1, h)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.conv2(F.gelu(self.norm(self.conv1(x)))))


class TCN(nn.Module):
    """Non-causal dilated residual CNN; receptive field ~ +-5 s at 50 Hz."""

    def __init__(self, n_in, n_glob, h=64, layers=2, dropout=0.2, use_global=True):
        super().__init__()
        self.use_global = use_global
        self.inp = nn.Conv1d(n_in + (16 if use_global else 0), h, 1)
        self.glob = nn.Sequential(nn.Linear(n_glob, 16), nn.GELU()) if use_global else None
        self.blocks = nn.Sequential(*[ResBlock(h, d, dropout) for d in (1, 2, 4, 8, 16, 32, 64)])
        self.ctx = nn.Sequential(nn.Linear(h, h), nn.GELU())
        self.head = nn.Sequential(nn.Conv1d(2 * h, h, 1), nn.GELU(), nn.Conv1d(h, 20, 1))
        self.cls = nn.Linear(h, len(ACTIONS))

    def forward(self, x, g):
        z = x.transpose(1, 2)
        if self.use_global:
            z = torch.cat([z, self.glob(g)[:, :, None].expand(-1, -1, z.shape[2])], 1)
        o = self.blocks(self.inp(z))
        pooled = o.mean(2)
        c = self.ctx(pooled)[:, :, None].expand(-1, -1, o.shape[2])
        y = self.head(torch.cat([o, c], 1)).transpose(1, 2)
        return y, self.cls(pooled)


class TFM(nn.Module):
    """Conv front (local context) + Transformer encoder with banded (local) attention.

    Attention is limited to +-win frames (50 Hz), so training crops and full-length
    test recordings see the same attention span. No absolute positions: order comes
    from the conv front.
    """

    def __init__(self, n_in, n_glob, h=96, layers=3, dropout=0.2, use_global=True, heads=4, win=150):
        super().__init__()
        self.use_global = use_global
        self.win = win
        self.glob = nn.Sequential(nn.Linear(n_glob, 16), nn.GELU()) if use_global else None
        self.inp = nn.Conv1d(n_in + (16 if use_global else 0), h, 1)
        self.front = nn.Sequential(*[ResBlock(h, d, dropout) for d in (1, 2, 4)])
        layer = nn.TransformerEncoderLayer(h, heads, 2 * h, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(h)
        self.ctx = nn.Sequential(nn.Linear(h, h), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 20))
        self.cls = nn.Linear(h, len(ACTIONS))

    def forward(self, x, g):
        z = x.transpose(1, 2)
        if self.use_global:
            z = torch.cat([z, self.glob(g)[:, :, None].expand(-1, -1, z.shape[2])], 1)
        z = self.front(self.inp(z)).transpose(1, 2)  # (B, T, h)
        T = z.shape[1]
        mask = None
        if self.win > 0:
            idx = torch.arange(T, device=z.device)
            mask = (idx[None, :] - idx[:, None]).abs() > self.win  # True = not allowed
        o = self.norm(self.enc(z, mask=mask))
        pooled = o.mean(1)
        c = self.ctx(pooled)[:, None, :].expand(-1, T, -1)
        return self.head(torch.cat([o, c], -1)), self.cls(pooled)


def build_net(a: dict, n_in: int, n_glob: int):
    """Build any architecture from an args dict (also used by predict.py)."""
    arch = a.get("arch", "gru")
    common_args = (n_in, n_glob, a["hidden"], a["layers"], a["dropout"], not a["no_global"])
    if arch == "tfm":
        return TFM(*common_args, heads=a.get("heads", 4), win=a.get("win", 150))
    return {"gru": Net, "tcn": TCN}[arch](*common_args)


# ----------------------------------------------------------------------------- training
def make_batch(items, rng, crop, stretch, feat_noise, glob_noise):
    xs, gs, ys, ws, acts = [], [], [], [], []
    for fr, gl, yt, w, a in items:
        s = rng.uniform(1 / stretch, stretch) if stretch > 1 else 1.0
        L = min(int(round(crop * s)), fr.shape[0])
        st = rng.integers(0, fr.shape[0] - L + 1)
        xf = torch.from_numpy(fr[st : st + L]).T[None]
        yf = torch.from_numpy(yt[st : st + L]).T[None]
        xf = F.interpolate(xf, size=crop, mode="linear", align_corners=True)[0].T
        yf = F.interpolate(yf, size=crop, mode="linear", align_corners=True)[0].T
        if feat_noise > 0:
            xf = xf + feat_noise * torch.randn_like(xf)
        g = torch.from_numpy(gl) + glob_noise * torch.randn(len(gl))
        xs.append(xf); ys.append(yf); gs.append(g); ws.append(torch.from_numpy(w)); acts.append(a)
    return torch.stack(xs), torch.stack(gs), torch.stack(ys), torch.stack(ws), torch.tensor(acts)


def train_model(train_items, args, seed, val_fn=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_in = train_items[0][0].shape[1]
    n_glob = len(train_items[0][1])
    model = build_net(vars(args), n_in, n_glob)
    dev = torch.device(getattr(args, "device_resolved", "cpu"))
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    history = []
    for ep in range(args.epochs):
        model.train()
        tl = 0.0
        for _ in range(args.steps):
            pick = rng.integers(0, len(train_items), args.batch)
            x, g, y, w, a = make_batch([train_items[i] for i in pick], rng, args.crop, args.stretch,
                                       args.feat_noise, args.glob_noise)
            x, g, y, w, a = x.to(dev), g.to(dev), y.to(dev), w.to(dev), a.to(dev)
            pred, logits = model(x, g)
            loss = ((pred - y).abs() * w[:, None, :]).sum(-1).mean()
            loss = loss + args.cls_w * F.cross_entropy(logits, a)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item()
        if val_fn is not None and ((ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs):
            v = val_fn(model)
            history.append((ep + 1, v))
            print(f"    ep {ep + 1:3d} train {tl / args.steps:.4f} val {v:.4f}", flush=True)
    return model, history


@torch.no_grad()
def predict(model, fr, gl, T):
    model.eval()
    dev = next(model.parameters()).device
    y, logits = model(torch.from_numpy(fr)[None].to(dev), torch.from_numpy(gl)[None].to(dev))
    return upsample(y[0].cpu().numpy(), T), torch.softmax(logits[0], -1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="gru", choices=["gru", "tcn", "tfm"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=600)
    ap.add_argument("--stretch", type=float, default=1.25)
    ap.add_argument("--feat-noise", type=float, default=0.1)
    ap.add_argument("--glob-noise", type=float, default=0.3)
    ap.add_argument("--cls-w", type=float, default=0.1)
    ap.add_argument("--no-global", action="store_true")
    ap.add_argument("--heads", type=int, default=4, help="tfm: attention heads")
    ap.add_argument("--win", type=int, default=150, help="tfm: attention window +-frames (0 = full)")
    ap.add_argument("--shuffle-targets", action="store_true",
                    help="SANITY CHECK: give each training recording the targets of another one")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto = cuda if available, else cpu")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "auto":
        args.device_resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        args.device_resolved = args.device
    print(f"device: {args.device_resolved}", flush=True)
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)

    recs = load_train()
    fr, gl = featurize(recs, "train")
    gmat = np.stack(gl)

    def items_for(idx, gmean, gstd):
        return [(fr[i], (gl[i] - gmean) / gstd, frame_targets(recs[i].y), angle_weights(recs[i].action),
                 ACTIONS.index(recs[i].action)) for i in idx]

    res = {"args": vars(args)}
    if not args.skip_cv:
        folds = make_folds(recs)
        oof, oof_cls = {}, {}
        t0 = time.time()
        for k, f in enumerate(folds):
            tr = [i for i, r in enumerate(recs) if r.session in f["train_sessions"]]
            va = [i for i, r in enumerate(recs) if r.session in f["val_sessions"]]
            gmean, gstd = gmat[tr].mean(0), gmat[tr].std(0) + 1e-6
            items = items_for(tr, gmean, gstd)
            if args.shuffle_targets:
                perm = np.random.default_rng(123 + k).permutation(len(items))
                mixed = []
                for i, j in enumerate(perm):
                    L = min(items[i][0].shape[0], items[j][2].shape[0])
                    mixed.append((items[i][0][:L], items[i][1], items[j][2][:L], items[j][3], items[j][4]))
                items = mixed
                print(f"  fold {k}: targets shuffled between training recordings", flush=True)
            fold_preds = {recs[i].rid: [] for i in va}
            fold_cls = {recs[i].rid: [] for i in va}

            def val_fn(model, va=va, gmean=gmean, gstd=gstd):
                p = {recs[i].rid: predict(model, fr[i], (gl[i] - gmean) / gstd, len(recs[i].x))[0] for i in va}
                return cv_score([recs[i] for i in va], p)["cv"]

            for s in range(args.seeds):
                print(f"fold {k} seed {s}", flush=True)
                model, hist = train_model(items, args, 1000 * k + s, val_fn if s == 0 else None)
                for i in va:
                    p, c = predict(model, fr[i], (gl[i] - gmean) / gstd, len(recs[i].x))
                    fold_preds[recs[i].rid].append(p)
                    fold_cls[recs[i].rid].append(c)
            for rid in fold_preds:
                oof[rid] = np.mean(fold_preds[rid], 0)
                oof_cls[rid] = np.mean(fold_cls[rid], 0)
        sc = cv_score(recs, oof, verbose=True)
        acc = np.mean([ACTIONS[int(np.argmax(oof_cls[r.rid]))] == r.action for r in recs])
        print(f"aux action accuracy {acc:.3f}   time {time.time() - t0:.0f}s")
        res.update(sc)
        res["cls_acc"] = float(acc)
        np.savez_compressed(out / "oof.npz", **{rid: v.astype(np.float32) for rid, v in oof.items()})
        np.savez_compressed(out / "oof_cls.npz", **oof_cls)

    if args.final:
        gmean, gstd = gmat.mean(0), gmat.std(0) + 1e-6
        items = items_for(range(len(recs)), gmean, gstd)
        tests = {d: load_test(d) for d in ("phase1_test", "test") if (ROOT / "data" / d).exists()}
        tfeat = {d: featurize(tr, d) for d, tr in tests.items()}
        preds = {d: {} for d in tests}
        for s in range(args.seeds):
            print(f"final seed {s}", flush=True)
            model, _ = train_model(items, args, 777 + s)
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / f"final_seed{s}.pt")
            for d, trs in tests.items():
                tfr, tgl = tfeat[d]
                for j, r in enumerate(trs):
                    p, _ = predict(model, tfr[j], (tgl[j] - gmean) / gstd, len(r.x))
                    preds[d].setdefault(r.rid, []).append(p)
        for d in preds:
            np.savez_compressed(out / f"pred_{d}.npz",
                                **{rid: np.mean(v, 0).astype(np.float32) for rid, v in preds[d].items()})
        np.savez(out / "glob_norm.npz", mean=gmean, std=gstd)
    (out / ("result.json" if not args.skip_cv else "result_final.json")).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
