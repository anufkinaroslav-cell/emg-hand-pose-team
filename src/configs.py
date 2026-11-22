"""Final solution: the four models of the ensemble and their weights.

Every model is trained with 3 seeds; the ensemble averages the seed-averaged
predictions of the four models with the weights below.
Day-disjoint CV of this ensemble: Weighted MAE 0.1058 (see docs/EXPERIMENTS.md).
"""
from __future__ import annotations

SEEDS = 3

# name -> (ensemble weight, nn.py arguments)
ENSEMBLE: dict[str, tuple[float, list[str]]] = {
    "tcn_h128": (2.0, ["--arch", "tcn", "--hidden", "128", "--dropout", "0.3"]),
    "gru_h128l3": (1.0, ["--arch", "gru", "--crop", "300", "--batch", "24", "--hidden", "128", "--layers", "3"]),
    "tcn_h128e60": (1.0, ["--arch", "tcn", "--hidden", "128", "--dropout", "0.3", "--epochs", "60"]),
    "gru_h96l3": (1.0, ["--arch", "gru", "--crop", "300", "--batch", "24", "--hidden", "96", "--layers", "3"]),
}

# Sanity check: same TCN, but every training recording gets the targets of another one.
SANITY_SHUFFLE = ["--arch", "tcn", "--shuffle-targets", "--seeds", "1"]
