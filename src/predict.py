"""Predict a test directory with saved final models (no retraining).

    python src/predict.py --models tcn_v1 [gru_c300 ...] --input-dir test \
        --template data/phase2_sample_submission.csv --output outputs/submission_phase2.csv

Each model directory outputs/<name>/ (or models/<name>/, committed to git) must contain final_seed*.pt, glob_norm.npz
and result*.json (its training arguments). Predictions of all seeds and all
listed models are averaged (optionally weighted with --weights).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
import torch

from common import ROOT, load_test
from nn import OUT, build_net, featurize, predict


def model_args(d):
    for name in ("result_final.json", "result.json"):
        if (d / name).exists():
            return json.loads((d / name).read_text())["args"]
    raise FileNotFoundError(f"no result*.json in {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--input-dir", default="test", help="directory under data/")
    ap.add_argument("--template", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)

    recs = load_test(args.input_dir)
    assert recs, f"no recordings in data/{args.input_dir}"
    fr, gl = featurize(recs, args.input_dir)
    # Sanity check: train EMG is in volts (std ~3e-5). A test set in other units would push the
    # global loudness features far outside the training range.
    level = np.median([np.std(r.x) for r in recs])
    if not 5e-6 < level < 2e-4:
        print(f"WARNING: test EMG std ~{level:.2e}, train ~3e-5. Different units? "
              "Global loudness features will be out of range; check the data before submitting.")
    weights = args.weights or [1.0] * len(args.models)
    packs = []
    for name in args.models:
        d = OUT / name if (OUT / name / "glob_norm.npz").exists() else ROOT / "models" / name
        a = model_args(d)
        norm = np.load(d / "glob_norm.npz")
        ckpts = sorted(d.glob("final_seed*.pt"))
        assert ckpts, f"no final_seed*.pt in {d}"
        preds = {}
        for c in ckpts:
            m = build_net(a, fr[0].shape[1], len(gl[0]))
            m.load_state_dict(torch.load(c, map_location="cpu"))
            for r, f, g in zip(recs, fr, gl):
                p, _ = predict(m, f, ((g - norm["mean"]) / norm["std"]).astype(np.float32), len(r.x))
                preds.setdefault(r.rid, []).append(p)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        path = OUT / name / f"pred_{args.input_dir}.npz"
        np.savez_compressed(path, **{k: np.mean(v, 0).astype(np.float32) for k, v in preds.items()})
        packs.append(str(path))
        print(f"{name}: {len(ckpts)} checkpoints -> {path}")

    cmd = [sys.executable, str(ROOT / "src" / "make_submission.py"), "--pred", *packs,
           "--weights", *map(str, weights), "--template", args.template, "--output", args.output]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
