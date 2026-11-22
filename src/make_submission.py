"""Write a submission CSV from saved per-recording predictions.

    python src/make_submission.py --pred outputs/gru_v1/pred_test.npz \
        --template data/phase2_sample_submission.csv --output outputs/submission_phase2.csv

Several --pred files are averaged (optionally with --weights).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

ANGLE_COLS = [f"angle_{i:02d}" for i in range(20)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--template", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    weights = args.weights or [1.0] * len(args.pred)
    assert len(weights) == len(args.pred)
    packs = [np.load(p) for p in args.pred]
    tpl = pd.read_csv(args.template)
    assert list(tpl.columns) == ["row_id", "recording_id", "sample_index"] + ANGLE_COLS, "unexpected template columns"

    out = np.empty((len(tpl), 20), dtype=np.float64)
    for rid, idx in tpl.groupby("recording_id", sort=False).indices.items():
        p = sum(w * pk[rid] for w, pk in zip(weights, packs)) / sum(weights)
        si = tpl.sample_index.values[idx]
        assert si.max() < len(p), f"{rid}: sample_index beyond recording length"
        out[idx] = p[si]
    out = np.clip(out, -np.pi, np.pi)
    assert np.isfinite(out).all()

    sub = tpl[["row_id", "recording_id", "sample_index"]].copy()
    sub[ANGLE_COLS] = out.astype(np.float32)
    sub.to_csv(args.output, index=False, float_format="%.6g")
    print(f"wrote {args.output}: {len(sub)} rows, {sub.recording_id.nunique()} recordings")


if __name__ == "__main__":
    main()
