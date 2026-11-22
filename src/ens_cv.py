"""Score single models and their ensemble on saved out-of-fold predictions.

    cd src && python ens_cv.py tcn_v1 tcn_h96 gru_c300
    cd src && python ens_cv.py tcn_v1 tcn_h96 gru_c300 --weights 1 1 2

Each name must have outputs/<name>/oof.npz (written by `python nn.py --name <name>` CV run).
"""
from __future__ import annotations

import argparse

import numpy as np

from common import cv_score, load_train
from nn import OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--weights", nargs="+", type=float)
    args = ap.parse_args()
    weights = args.weights or [1.0] * len(args.names)
    assert len(weights) == len(args.names), "one weight per model"

    recs = load_train()
    packs = []
    for name in args.names:
        path = OUT / name / "oof.npz"
        assert path.exists(), f"нет {path}: сначала запустите CV для {name}"
        pk = np.load(path)
        packs.append(pk)
        score = cv_score(recs, {r.rid: pk[r.rid] for r in recs})["cv"]
        print(f"{name:<20} CV {score:.4f}")

    ens = {r.rid: sum(w * pk[r.rid] for w, pk in zip(weights, packs)) / sum(weights) for r in recs}
    print("-" * 40)
    print("Ансамбль", " + ".join(args.names), "веса", weights)
    cv_score(recs, ens, verbose=True)


if __name__ == "__main__":
    main()
