"""Find the best ensemble on out-of-fold predictions (greedy forward selection, with replacement).

    cd src && python ens_search.py tcn_v1_s3 tcn_h96_s3 gru_c300_s3 tfm_a_s3 ...

Prints every model's CV, the equal-weight ensemble of all, and the greedy ensemble
(Caruana et al.): repeatedly add the model that lowers CV the most. Weights = how many
times a model was picked. Greedy selection is tuned on the same CV, so treat small gains
(< 0.001) as noise.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from common import cv_score, load_train
from nn import OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--rounds", type=int, default=12)
    args = ap.parse_args()

    recs = load_train()
    oof = {}
    for n in args.names:
        path = OUT / n / "oof.npz"
        if not path.exists():
            print(f"skip {n}: no oof.npz")
            continue
        z = np.load(path)
        oof[n] = {r.rid: z[r.rid] for r in recs}
    names = list(oof)

    def score(counter):
        tot = sum(counter.values())
        pred = {r.rid: sum(c * oof[n][r.rid] for n, c in counter.items()) / tot for r in recs}
        return cv_score(recs, pred)["cv"]

    print("Single models:")
    for n in sorted(names, key=lambda n: score(Counter({n: 1}))):
        print(f"  {n:<20} {score(Counter({n: 1})):.4f}")
    print(f"Equal weights, all {len(names)}: {score(Counter({n: 1 for n in names})):.4f}")

    chosen, best = Counter(), 9.0
    for _ in range(args.rounds):
        cand = min(names, key=lambda n: score(chosen + Counter({n: 1})))
        s = score(chosen + Counter({cand: 1}))
        if s >= best - 1e-5:
            break
        chosen[cand] += 1
        best = s
        print(f"  + {cand:<20} -> {best:.4f}")
    print("Greedy ensemble:", dict(chosen), f"CV {best:.4f}")
    tot = sum(chosen.values())
    pred = {r.rid: sum(c * oof[n][r.rid] for n, c in chosen.items()) / tot for r in recs}
    cv_score(recs, pred, verbose=True)


if __name__ == "__main__":
    main()
