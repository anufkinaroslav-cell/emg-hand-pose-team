"""One entry point for the whole solution.

    python reproduce.py check      # data is in place + "always mean pose" baseline (CV ~0.377)
    python reproduce.py cv         # day-disjoint CV of the 4 ensemble models + the ensemble (~0.106)
    python reproduce.py sanity     # leakage check: shuffled targets must give a bad score (~0.47)
    python reproduce.py final      # train the 4 models on all 80 recordings (3 seeds each)
    python reproduce.py predict --input-dir test --template data/phase2_sample_submission.csv --output outputs/submission_phase2.csv

Options for cv / final / sanity: --device auto|cpu|cuda (default auto), --threads N,
--smoke (2 tiny epochs, outputs get a "_smoke" suffix: checks that the pipeline runs, scores are meaningless).
Finished steps are skipped, so every command can simply be re-run after an interruption.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
OUT = ROOT / "outputs"
sys.path.insert(0, str(SRC))

from configs import ENSEMBLE, SANITY_SHUFFLE, SEEDS  # noqa: E402


def run(cmd: list[str], cwd: Path = SRC) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], cwd=cwd, check=True)


def cmd_check(_args) -> None:
    import numpy as np

    from common import cv_score, load_train, make_folds

    recs = load_train()
    days = {r.day for r in recs}
    sessions = {r.session for r in recs}
    print(f"train recordings: {len(recs)}, days: {len(days)}, sessions: {len(sessions)}")
    print(f"signal shape {recs[0].x.shape}, target shape {recs[0].y.shape}")
    preds = {}
    for f in make_folds(recs):
        tr = [r for r in recs if r.session in f["train_sessions"]]
        mean_pose = np.concatenate([r.y for r in tr]).mean(0)
        for r in recs:
            if r.session in f["val_sessions"]:
                preds[r.rid] = np.repeat(mean_pose[None], len(r.y), 0)
    print("baseline 'always the mean pose':")
    cv_score(recs, preds, verbose=True)


SMOKE_ARGS = ["--epochs", "2", "--steps", "2", "--eval-every", "1"]


def tag(args) -> str:
    return "_smoke" if getattr(args, "smoke", False) else ""


def fast(args) -> list[str]:
    return SMOKE_ARGS if getattr(args, "smoke", False) else []


def cmd_cv(args) -> None:
    for name, (_, extra) in ENSEMBLE.items():
        cv_name = f"{name}_cv{tag(args)}"
        if (OUT / cv_name / "oof.npz").exists():
            print(f"skip {cv_name}: already done", flush=True)
            continue
        run(["nn.py", "--name", cv_name, "--seeds", str(SEEDS), "--device", args.device,
             "--threads", str(args.threads), *extra, *fast(args)])
    names = [f"{n}_cv{tag(args)}" for n in ENSEMBLE]
    weights = [str(w) for w, _ in ENSEMBLE.values()]
    run(["ens_cv.py", *names, "--weights", *weights])


def cmd_sanity(args) -> None:
    run(["nn.py", "--name", f"sanity_shuffle{tag(args)}", "--device", args.device, "--threads", str(args.threads),
         *SANITY_SHUFFLE, *fast(args)])
    print("On the competition data expected: CV WORSE than the mean pose (0.377), e.g. ~0.47,\n"
          "and action accuracy ~0.125 (chance). A good score here would mean a leak.")


def cmd_final(args) -> None:
    for name, (_, extra) in ENSEMBLE.items():
        out = f"final_{name}{tag(args)}"
        if len(list((OUT / out).glob("final_seed*.pt"))) >= SEEDS:
            print(f"skip {out}: already trained", flush=True)
            continue
        run(["nn.py", "--name", out, "--skip-cv", "--final", "--seeds", str(SEEDS), "--device", args.device,
             "--threads", str(args.threads), *extra, *fast(args)])


def cmd_predict(args) -> None:
    names = [f"final_{n}{tag(args)}" for n in ENSEMBLE]
    weights = [str(w) for w, _ in ENSEMBLE.values()]
    run(["src/predict.py", "--models", *names, "--weights", *weights, "--input-dir", args.input_dir,
         "--template", args.template, "--output", args.output], cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    for name in ("cv", "sanity", "final"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline works")
    p = sub.add_parser("predict")
    p.add_argument("--smoke", action="store_true", help="use the *_smoke models")
    p.add_argument("--input-dir", default="test", help="folder under data/ with rec_*.npy")
    p.add_argument("--template", default="data/phase2_sample_submission.csv")
    p.add_argument("--output", default="outputs/submission_phase2.csv")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"check": cmd_check, "cv": cmd_cv, "sanity": cmd_sanity, "final": cmd_final, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
