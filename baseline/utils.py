"""Data loading, the session split, and the validation metric.

The training files under data/ are only read. Nothing here rewrites,
filters, or resamples the stored arrays.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


# Samples 0..999 are context. Loss, validation, and submissions use 1000..T-1.
CONTEXT_SAMPLES = 1000
N_CHANNELS = 6
N_ANGLES = 20

# Column order in the target files and in angle_info.json.
# Little comes before Ring.
ANGLE_NAMES = (
    "index_mcp_theta",
    "index_mcp_fi",
    "index_pip",
    "index_dip",
    "middle_mcp_theta",
    "middle_mcp_fi",
    "middle_pip",
    "middle_dip",
    "little_mcp_theta",
    "little_mcp_fi",
    "little_pip",
    "little_dip",
    "ring_mcp_theta",
    "ring_mcp_fi",
    "ring_pip",
    "ring_dip",
    "thumb_mcp_theta",
    "thumb_mcp_fi",
    "thumb_pip",
    "thumb_dip",
)

# This order is the angle-column order, not alphabetical order.
FINGER_NAMES = ("Index", "Middle", "Little", "Ring", "Thumb")
FINGER_SLICES = {name: (4 * index, 4 * index + 4) for index, name in enumerate(FINGER_NAMES)}
WHOLE_HAND_ACTIONS = ("Fist", "Open", "Close")
KNOWN_ACTIONS = frozenset(FINGER_NAMES) | frozenset(WHOLE_HAND_ACTIONS)

METADATA_COLUMNS = ["recording_id", "day_id", "session_id", "action", "n_samples"]
SUBMISSION_COLUMNS = (
    ["row_id", "recording_id", "sample_index"]
    + [f"angle_{index:02d}" for index in range(N_ANGLES)]
)

RECORDING_ID_RE = re.compile(r"rec_[0-9a-f]{16}\Z")

SPLIT_RULE = (
    "Sessions from the same day stay together. "
    "The day with three sessions is one validation fold. "
    "The day with two sessions is joined with one single-session day. "
    "The five single-session days are sorted, shuffled with NumPy's "
    "legacy RandomState(seed), and used in that order: the first joins "
    "the two-session day, and the other four form two pairs. "
    "Each session is held out once."
)


@dataclass(eq=False)
class Recording:
    recording_id: str
    day_id: str
    session_id: str
    action: str
    signal: np.ndarray  # float32, shape (T, 6)
    target: np.ndarray | None = None  # float32, shape (T, 20)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch.

    The same seed makes a run repeatable on one device. CPU, CUDA, and
    MPS can still differ slightly in floating-point results.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """Resolve auto, cpu, cuda, or mps.

    auto uses CUDA if it exists, otherwise MPS, otherwise CPU.
    An explicit device that is not available is an error.
    """
    choice = name.lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but CUDA is not available.")
        return torch.device("cuda")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps' but MPS is not available.")
        return torch.device("mps")
    raise ValueError(f"Unknown device {name!r}. Choose auto, cpu, cuda, or mps.")


def resolve_path(path) -> Path:
    """Absolute path with symlinks and '..' resolved. The path need not exist."""
    return Path(path).expanduser().resolve()


def ensure_output_outside_data(output, data_roots) -> Path:
    """Return a resolved output path that is not inside a data directory.

    Call this before creating directories or opening output files.
    """
    output = resolve_path(output)
    for root in data_roots:
        root = resolve_path(root)
        if output == root or root in output.parents:
            raise ValueError(
                f"Refusing to write {output} inside the data directory {root}. "
                "Choose an output path outside the competition data."
            )
    return output


def competition_data_root(input_dir) -> Path | None:
    """Parent of input_dir when that parent is the public competition data tree.

    The participant layout keeps train/, phase1_test/, and test/ together,
    next to the sample submission files. A synthetic folder without those
    markers has no extra protected parent.
    """
    parent = resolve_path(input_dir).parent
    markers = (
        parent / "train",
        parent / "phase1_sample_submission.csv",
        parent / "phase2_sample_submission.csv",
    )
    if any(marker.exists() for marker in markers):
        return parent
    return None


def load_train_recordings(data_dir) -> list[Recording]:
    """Load data_dir/train. Arrays are converted to float32 for PyTorch."""
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    metadata_path = train_dir / "metadata.csv"
    _check_angle_info(train_dir / "angle_info.json")
    rows = _read_metadata(metadata_path)
    signal_dir = train_dir / "signals"
    target_dir = train_dir / "targets"
    _check_id_sets(signal_dir, rows)
    _check_id_sets(target_dir, rows)

    session_days: dict[str, str] = {}
    recordings = []
    for row in rows:
        previous = session_days.setdefault(row["session_id"], row["day_id"])
        if previous != row["day_id"]:
            raise ValueError(
                f"Session {row['session_id']} is listed under both {previous} and {row['day_id']}"
            )
        recording_id = row["recording_id"]
        signal = _load_array(signal_dir / f"{recording_id}.npy", N_CHANNELS, row["n_samples"])
        target = _load_array(target_dir / f"{recording_id}.npy", N_ANGLES, row["n_samples"])
        recordings.append(
            Recording(
                recording_id=recording_id,
                day_id=row["day_id"],
                session_id=row["session_id"],
                action=row["action"],
                signal=signal,
                target=target,
            )
        )
    recordings.sort(key=lambda recording: recording.recording_id)
    return recordings


def fit_normalization(recordings: list[Recording]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and population standard deviation.

    Statistics are computed on every time sample of the recordings
    passed in. Callers pass only the training sessions for the current
    fit. The standard deviation divides by N, not N-1.
    """
    if not recordings:
        raise ValueError("Cannot fit normalization without recordings")
    pieces = []
    for recording in recordings:
        signal = np.asarray(recording.signal)
        if signal.ndim != 2 or signal.shape[1] != N_CHANNELS:
            raise ValueError(
                f"{recording.recording_id} signal shape {signal.shape} is not (time, {N_CHANNELS})"
            )
        pieces.append(np.asarray(signal, dtype=np.float64))
    data = np.concatenate(pieces, axis=0)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(
            "Per-channel standard deviation must be positive and finite. "
            f"mean={mean.tolist()} std={std.tolist()}"
        )
    return mean.astype(np.float32), std.astype(np.float32)


def make_cv_folds(recordings: list[Recording], seed: int) -> list[dict]:
    """Four day-disjoint folds. Validation sizes are 3, 3, 2, and 2 sessions."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    session_day: dict[str, str] = {}
    session_actions: dict[str, list[str]] = defaultdict(list)
    for recording in recordings:
        if not recording.session_id or not recording.day_id:
            raise ValueError(f"{recording.recording_id} is missing a session id or day id")
        previous = session_day.setdefault(recording.session_id, recording.day_id)
        if previous != recording.day_id:
            raise ValueError(
                f"Session {recording.session_id} is assigned to both {previous} and {recording.day_id}"
            )
        session_actions[recording.session_id].append(recording.action)

    expected_actions = sorted(KNOWN_ACTIONS)
    for session_id, actions in session_actions.items():
        if sorted(actions) != expected_actions:
            raise ValueError(
                f"Session {session_id} must contain exactly one recording of each action "
                f"{expected_actions}. Found {sorted(actions)}."
            )

    day_to_sessions: dict[str, list[str]] = defaultdict(list)
    for session_id, day_id in session_day.items():
        day_to_sessions[day_id].append(session_id)
    for day_id in day_to_sessions:
        day_to_sessions[day_id] = sorted(day_to_sessions[day_id])

    days_by_size: dict[int, list[str]] = defaultdict(list)
    for day_id, sessions in day_to_sessions.items():
        days_by_size[len(sessions)].append(day_id)
    if (
        sorted(days_by_size) != [1, 2, 3]
        or len(days_by_size[1]) != 5
        or len(days_by_size[2]) != 1
        or len(days_by_size[3]) != 1
    ):
        found = {day: sessions for day, sessions in sorted(day_to_sessions.items())}
        raise ValueError(
            "Expected 5 days with one session, 1 day with two sessions, "
            f"and 1 day with three sessions. Found {found}."
        )

    # RandomState (not Generator) keeps this shuffle on NumPy's legacy generator.
    rng = np.random.RandomState(seed)
    single_days = sorted(days_by_size[1])
    rng.shuffle(single_days)
    triple = day_to_sessions[days_by_size[3][0]]
    double = day_to_sessions[days_by_size[2][0]]
    groups = [
        tuple(triple),
        tuple(sorted(double + day_to_sessions[single_days[0]])),
        tuple(sorted(day_to_sessions[single_days[1]] + day_to_sessions[single_days[2]])),
        tuple(sorted(day_to_sessions[single_days[3]] + day_to_sessions[single_days[4]])),
    ]

    all_sessions = sorted(session_day)
    folds = []
    for validation_sessions in groups:
        validation_set = set(validation_sessions)
        training_sessions = tuple(session_id for session_id in all_sessions if session_id not in validation_set)
        folds.append(
            {
                "val_sessions": list(validation_sessions),
                "train_sessions": list(training_sessions),
                "val_days": sorted({session_day[session_id] for session_id in validation_sessions}),
                "train_days": sorted({session_day[session_id] for session_id in training_sessions}),
            }
        )
    _assert_folds(folds, session_day, session_actions)
    return folds


def recording_weighted_mae(prediction, target, action: str) -> float:
    """Weighted MAE of one recording, in radians, on samples 1000..T-1.

    An isolated finger action scores
        0.5 * MAE(active finger) + 0.5 * mean MAE(other four fingers).
    Fist, Open, and Close score the mean MAE of all five fingers.
    Each finger MAE is the mean absolute error over time and its four angles.
    Differences are ordinary absolute differences, with no angle wrapping.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != N_ANGLES:
        raise ValueError(
            f"Prediction and target must both have shape (time, {N_ANGLES}), "
            f"got {prediction.shape} and {target.shape}"
        )
    if action not in KNOWN_ACTIONS:
        raise ValueError(f"Unknown action {action!r}")
    if prediction.shape[0] <= CONTEXT_SAMPLES:
        raise ValueError(
            f"Recording length {prediction.shape[0]} does not extend past "
            f"the {CONTEXT_SAMPLES}-sample context"
        )
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite values in a prediction or target")

    scored_prediction = prediction[CONTEXT_SAMPLES:]
    scored_target = target[CONTEXT_SAMPLES:]
    finger_mae = []
    for finger in FINGER_NAMES:
        start, stop = FINGER_SLICES[finger]
        error = np.abs(scored_prediction[:, start:stop] - scored_target[:, start:stop])
        finger_mae.append(float(np.mean(error)))

    if action in WHOLE_HAND_ACTIONS:
        return float(np.mean(finger_mae))
    active = FINGER_NAMES.index(action)
    others = [value for index, value in enumerate(finger_mae) if index != active]
    return float(0.5 * finger_mae[active] + 0.5 * float(np.mean(others)))


def aggregate_session_scores(recording_scores: list[tuple[str, float]]) -> dict:
    """Average recordings within a session, then average sessions.

    A long recording does not outweigh a short one. Each recording score
    counts once, and each session score counts once.
    """
    if not recording_scores:
        raise ValueError("No recording scores to aggregate")
    grouped: dict[str, list[float]] = {}
    order: list[str] = []
    for session_id, score in recording_scores:
        if session_id not in grouped:
            grouped[session_id] = []
            order.append(session_id)
        grouped[session_id].append(float(score))
    sessions = []
    for session_id in order:
        scores = grouped[session_id]
        sessions.append(
            {
                "session_id": session_id,
                "weighted_mae": float(np.mean(scores)),
                "n_recordings": len(scores),
            }
        )
    overall = float(np.mean([session["weighted_mae"] for session in sessions]))
    return {"sessions": sessions, "weighted_mae": overall}


def pearson_by_angle(prediction, target) -> list:
    """Pearson r for each angle on samples 1000..T-1.

    None means the correlation is undefined because that prediction
    or that target does not vary. None is not a zero.
    """
    scored_prediction, scored_target = _scored_pair(prediction, target)
    return _pearson_columns(scored_prediction, scored_target)


def pearson_by_angle_concatenated(series) -> list:
    """Pearson r after joining scored samples in recording-id order.

    `series` is a list of (recording_id, prediction, target). Predictions
    and targets are full-length arrays and are sliced at the same index.
    """
    if not series:
        raise ValueError("No recordings to correlate")
    ordered = sorted(series, key=lambda item: item[0])
    predictions = []
    targets = []
    for _recording_id, prediction, target in ordered:
        scored_prediction, scored_target = _scored_pair(prediction, target)
        predictions.append(scored_prediction)
        targets.append(scored_target)
    return _pearson_columns(np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0))


def summarize_correlations(values) -> dict:
    """Mean of the defined correlations, and how many were undefined."""
    defined = []
    undefined = 0
    for value in values:
        if value is None:
            undefined += 1
            continue
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("A correlation was non-finite. Use None when Pearson is undefined.")
        defined.append(number)
    mean = float(np.mean(np.asarray(defined, dtype=np.float64))) if defined else None
    return {"mean": mean, "defined": len(defined), "undefined": undefined}


def _scored_pair(prediction, target) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != N_ANGLES:
        raise ValueError(
            f"Prediction and target must both have shape (time, {N_ANGLES}), "
            f"got {prediction.shape} and {target.shape}"
        )
    if prediction.shape[0] <= CONTEXT_SAMPLES:
        raise ValueError(
            f"Recording length {prediction.shape[0]} does not extend past "
            f"the {CONTEXT_SAMPLES}-sample context"
        )
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite values in a prediction or target")
    return prediction[CONTEXT_SAMPLES:], target[CONTEXT_SAMPLES:]


def _pearson_columns(prediction, target) -> list:
    correlations = []
    for angle in range(N_ANGLES):
        x = prediction[:, angle]
        y = target[:, angle]
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            correlations.append(None)
            continue
        value = float(np.corrcoef(x, y)[0, 1])
        if not math.isfinite(value):
            correlations.append(None)
        else:
            correlations.append(value)
    return correlations


def rounded_median(values) -> int:
    """Nearest integer to the median. Halves round away from zero (2.5 -> 3)."""
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("No values to summarize")
    median = float(np.median(np.asarray(numbers, dtype=np.float64)))
    if median >= 0:
        return int(np.floor(median + 0.5))
    return int(np.ceil(median - 0.5))


def batch_channels_first(array, n_features: int, device) -> torch.Tensor:
    """Convert a (time, features) array to a (1, features, time) float32 batch."""
    values = np.asarray(array)
    if values.ndim != 2 or values.shape[1] != n_features:
        raise ValueError(f"Expected an array of shape (time, {n_features}), got {values.shape}")
    channels_first = np.ascontiguousarray(values.transpose(1, 0), dtype=np.float32)
    return torch.from_numpy(channels_first).unsqueeze(0).to(device)


def predict_recording(model, signal, device) -> np.ndarray:
    """Run one recording. Returns float32 predictions of shape (time, 20)."""
    signal = np.asarray(signal)
    if signal.ndim != 2 or signal.shape[1] != N_CHANNELS:
        raise ValueError(f"Expected a signal of shape (time, {N_CHANNELS}), got {signal.shape}")
    model.eval()
    model_input = batch_channels_first(signal, N_CHANNELS, device)
    with torch.no_grad():
        prediction = model(model_input)
    expected = (1, N_ANGLES, signal.shape[0])
    if tuple(prediction.shape) != expected:
        raise RuntimeError(f"Model returned shape {tuple(prediction.shape)}, expected {expected}")
    time_major = prediction.squeeze(0).detach().to("cpu").numpy().T
    return np.ascontiguousarray(time_major, dtype=np.float32)


def _check_angle_info(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing angle info: {path}")
    with path.open(encoding="utf-8") as handle:
        info = json.load(handle)
    if info.get("unit") != "rad":
        raise ValueError(f"{path} unit must be 'rad', found {info.get('unit')!r}")
    if info.get("angle_names") != list(ANGLE_NAMES):
        raise ValueError(f"{path} angle order does not match the expected 20 joint angles")


def _read_metadata(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != METADATA_COLUMNS:
            raise ValueError(f"{path} columns must be {','.join(METADATA_COLUMNS)}")
        rows = []
        seen = set()
        for line_no, row in enumerate(reader, start=2):
            recording_id = row["recording_id"]
            if not RECORDING_ID_RE.fullmatch(recording_id or ""):
                raise ValueError(f"{path}:{line_no} has invalid recording_id {recording_id!r}")
            if recording_id in seen:
                raise ValueError(f"Duplicate recording_id {recording_id}")
            seen.add(recording_id)
            action = row["action"]
            if action not in KNOWN_ACTIONS:
                raise ValueError(f"{path}:{line_no} has unknown action {action!r}")
            day_id = (row["day_id"] or "").strip()
            session_id = (row["session_id"] or "").strip()
            if not day_id or not session_id:
                raise ValueError(f"{path}:{line_no} is missing day_id or session_id")
            try:
                n_samples = int(row["n_samples"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no} has n_samples {row['n_samples']!r}") from exc
            if n_samples <= CONTEXT_SAMPLES:
                raise ValueError(
                    f"{recording_id} has n_samples={n_samples}, which leaves no scored "
                    f"samples after the {CONTEXT_SAMPLES}-sample context"
                )
            rows.append(
                {
                    "recording_id": recording_id,
                    "day_id": day_id,
                    "session_id": session_id,
                    "action": action,
                    "n_samples": n_samples,
                }
            )
    if not rows:
        raise ValueError(f"{path} contains no recordings")
    return rows


def _check_id_sets(directory: Path, rows: list[dict]) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    on_disk = {path.stem for path in directory.glob("*.npy")}
    in_metadata = {row["recording_id"] for row in rows}
    missing = sorted(in_metadata - on_disk)
    extra = sorted(on_disk - in_metadata)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing[:3]}")
        if extra:
            details.append(f"unexpected {extra[:3]}")
        raise ValueError(f"{directory} does not match metadata.csv ({'; '.join(details)})")


def _load_array(path: Path, n_features: int, n_samples: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    array = np.load(path)
    expected = (n_samples, n_features)
    if getattr(array, "shape", None) != expected:
        raise ValueError(f"{path} has shape {getattr(array, 'shape', None)}, expected {expected}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{path} has dtype {array.dtype}, expected a floating-point array")
    if not np.isfinite(array).all():
        raise ValueError(f"{path} contains non-finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _assert_folds(folds, session_day, session_actions) -> None:
    assert len(folds) == 4, len(folds)
    validation_ids = []
    all_sessions = set(session_day)
    for fold in folds:
        validation = list(fold["val_sessions"])
        training = list(fold["train_sessions"])
        assert len(validation) in (2, 3), validation
        assert len(training) == len(all_sessions) - len(validation), training
        assert set(validation).isdisjoint(training)
        assert set(validation) | set(training) == all_sessions
        validation_days = {session_day[session_id] for session_id in validation}
        training_days = {session_day[session_id] for session_id in training}
        assert validation_days.isdisjoint(training_days)
        assert fold["val_days"] == sorted(validation_days)
        assert fold["train_days"] == sorted(training_days)
        for part in (validation, training):
            actions = set()
            for session_id in part:
                actions.update(session_actions[session_id])
            assert actions == set(KNOWN_ACTIONS)
        validation_ids.extend(validation)
    assert sorted(len(fold["val_sessions"]) for fold in folds) == [2, 2, 3, 3]
    assert sorted(len(fold["train_sessions"]) for fold in folds) == [7, 7, 8, 8]
    assert sorted(validation_ids) == sorted(all_sessions)
    day_sessions = defaultdict(set)
    for session_id, day_id in session_day.items():
        day_sessions[day_id].add(session_id)
    for group in day_sessions.values():
        if len(group) < 2:
            continue
        for fold in folds:
            validation = set(fold["val_sessions"])
            assert group <= validation or group.isdisjoint(validation)
