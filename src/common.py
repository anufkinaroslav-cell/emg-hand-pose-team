"""Shared data loading, day-disjoint CV folds and the competition metric.

Folds and metric match the organizers' baseline (baseline/utils.py):
4 folds, sessions of one day stay together, CV score = mean of the
10 held-out session scores, each session = mean of its 8 recording scores.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "baseline"))

CONTEXT = 1000
FS = 500
FINGERS = ("Index", "Middle", "Little", "Ring", "Thumb")  # angle-column order
WHOLE = ("Fist", "Open", "Close")
ACTIONS = ("Index", "Middle", "Ring", "Little", "Thumb", "Fist", "Open", "Close")


@dataclass
class Rec:
    rid: str
    day: str
    session: str
    action: str
    x: np.ndarray  # (T, 6) float32
    y: np.ndarray | None  # (T, 20) float32


def load_train() -> list[Rec]:
    meta = pd.read_csv(DATA / "train" / "metadata.csv")
    recs = []
    for r in meta.itertuples():
        x = np.load(DATA / "train" / "signals" / f"{r.recording_id}.npy").astype(np.float32)
        y = np.load(DATA / "train" / "targets" / f"{r.recording_id}.npy").astype(np.float32)
        recs.append(Rec(r.recording_id, r.day_id, r.session_id, r.action, x, y))
    return recs


def load_test(dirname: str) -> list[Rec]:
    recs = []
    for f in sorted((DATA / dirname).glob("rec_*.npy")):
        x = np.load(f).astype(np.float32)
        recs.append(Rec(f.stem, "", "", "", x, None))
    return recs


def make_folds(recs: list[Rec], seed: int = 42) -> list[dict]:
    """Same folds as the baseline (seed 42)."""
    from utils import Recording, make_cv_folds  # baseline/utils.py

    shim = [Recording(r.rid, r.day, r.session, r.action, None, None) for r in recs]
    return make_cv_folds(shim, seed)


def wmae(pred: np.ndarray, target: np.ndarray, action: str) -> float:
    p = pred[CONTEXT:]
    t = target[CONTEXT:]
    fm = [float(np.mean(np.abs(p[:, 4 * i : 4 * i + 4] - t[:, 4 * i : 4 * i + 4]))) for i in range(5)]
    if action in WHOLE:
        return float(np.mean(fm))
    a = FINGERS.index(action)
    return 0.5 * fm[a] + 0.5 * float(np.mean([v for i, v in enumerate(fm) if i != a]))


def cv_score(recs: list[Rec], preds: dict[str, np.ndarray], verbose: bool = False) -> dict:
    """Score out-of-fold predictions: mean over sessions of mean over recordings."""
    rows = []
    for r in recs:
        if r.rid in preds:
            rows.append((r.session, r.action, wmae(preds[r.rid], r.y, r.action)))
    df = pd.DataFrame(rows, columns=["session", "action", "score"])
    ses = df.groupby("session").score.mean()
    out = {"cv": float(ses.mean()), "by_action": df.groupby("action").score.mean().round(4).to_dict(),
           "by_session": ses.round(4).to_dict()}
    if verbose:
        print(f"CV weighted MAE = {out['cv']:.4f}")
        print("  by action :", out["by_action"])
        print("  by session:", out["by_session"])
    return out
