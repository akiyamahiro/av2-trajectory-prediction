#!/usr/bin/env python3
"""History-only Transformer for single-agent vehicle trajectory prediction.

A standalone, deterministic K=1 baseline for AV2-style motion forecasting data.
It uses only the focal vehicle's observed history; maps and neighboring agents
are not inputs. The data, model, training loop, evaluation, and visualization
are all implemented in this file. Read main() first to find the CLI commands.

Requirements
------------
Python >= 3.10, PyTorch >= 2.2, numpy >= 1.24, pandas >= 2.0,
pyarrow >= 12, and matplotlib >= 3.7. No AV2 SDK is required.
Install PyTorch for your CPU/CUDA setup separately:
    https://pytorch.org/get-started/locally/
Install the data and plotting dependencies:
    python -m pip install "numpy>=1.24" "pandas>=2.0" "pyarrow>=12" "matplotlib>=3.7"

Input layout (relative to --project-root, which defaults to the current directory)
-------------------------------------------------------------------------------
data/raw/train/<id>/scenario_<id>.parquet
data/raw/val/<id>/scenario_<id>.parquet
splits/val_cv100.txt                      # one validation scenario ID per line
reports/cv_val100.csv                    # optional previous CV baseline

Obtain the dataset separately. Preserve the original train/validation split.
The validation ID list is an explicit input: reuse it for every model comparison.
If the optional CV report exists, its IDs and scores must match the cached data.

Quick start (run these commands from the project root)
----------------------------------------------------
python history_only_transformer.py check-environment
python history_only_transformer.py prepare --train-count 5000
python history_only_transformer.py check-batch --device cuda
python history_only_transformer.py train --device cuda --run history_overfit16 --overfit 16 --batch-size 16 --epochs 300
python history_only_transformer.py visualize --device cuda --split train --checkpoint checkpoints/history_overfit16/best.pt
python history_only_transformer.py train --device cuda --run history_tf_v1 --epochs 50
python history_only_transformer.py evaluate --device cuda --checkpoint checkpoints/history_tf_v1/best.pt
python history_only_transformer.py visualize --device cuda --checkpoint checkpoints/history_tf_v1/best.pt

Use --device cpu for CPU execution. Every subcommand has its own --help.
Existing history_v1 caches and model checkpoints remain compatible.
Skip prepare if the cache already exists; use a new --cache-name for new data.
Use a new --run for each experiment. Resume an interrupted run with:
    python history_only_transformer.py train --resume checkpoints/history_tf_v1/last.pt --batch-size 64 --epochs 100
Here --epochs is the TOTAL target epoch count. Resume uses the saved optimizer
learning rate unless --resume-lr is explicitly supplied.

Data and model contract
-----------------------
Each scene supplies one focal vehicle with complete timesteps 0..109.
Timesteps 0..49 are observed inputs; timesteps 50..109 are future labels.
The fixed local frame is defined by the position/heading at timestep 49.
Features: [local_x/50, local_y/50, local_vx/20, local_vy/20,
           sin(relative_heading), cos(relative_heading)].
Input: [B, 50, 6]. Output: [B, 60, 2] normalized local positions, not increments.
A linear projection and learned temporal embedding feed a Transformer encoder.
The final historical token feeds an MLP that predicts all 60 future points.
All historical tokens are observed, so history attention is bidirectional.
Future labels are never encoder inputs. Dataset masks use True for valid points;
PyTorch's padding mask uses True for ignored points, so the mask is inverted.

Loss is the mean squared Euclidean distance in normalized coordinates.
Evaluation restores meters: ADE is the mean future error, FDE is the final
future error, and MR is the fraction of scenes with FDE strictly greater than
2 meters. K=1, so ADE/FDE also equal minADE/minFDE for these predictions.

Outputs and scope
-----------------
prepare:   data/processed/<cache-name>/{train,val}.npz, ID lists, scan CSV, metadata
train:     checkpoints/<run>/{best,last}.pt, config.json, training.csv
evaluate:  reports/transformer_<run>.{csv,json}
visualize: figures/<run>/<id>.png (training plots use a train/ subdirectory)

Normal runs select best.pt by validation FDE. Overfit diagnostics select it by
training replay FDE and disable dropout. The same validation scenes are used
throughout; they are a development set, not an independent final test set.
This is a vehicle-only internal comparison, not a full official AV2 benchmark.
Missing frames, hidden test labels, multimodal predictions, maps, interactions,
and diffusion are outside the current scope. No trained accuracy is promised.
Share source code separately from raw data, generated caches, and checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import time

try:
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset
except ImportError as exc:
    raise SystemExit(
        f"Cannot load a required dependency: {exc}\n"
        f"Interpreter: {sys.executable}\n"
        "See the Requirements section at the top of this file."
    ) from exc


# ---------------------------------------------------------------------------
# 1. Data protocol, coordinate transforms, and cache helpers
# ---------------------------------------------------------------------------

VERSION = "history_v1"
POS_SCALE = 50.0
VEL_SCALE = 20.0
DT = 0.1


class RejectedSample(ValueError):
    """A valid file that does not meet our declared vehicle-only protocol."""


def encode_dataframe(df):
    """Encode one complete focal vehicle in a frame defined only by its history."""
    focal = df["focal_track_id"].iloc[0]
    target = df.loc[df["track_id"] == focal].sort_values("timestep")
    if target.empty:
        raise ValueError("focal_track_id does not match any track_id")
    kind = str(target["object_type"].iloc[0]).lower()
    if kind != "vehicle":
        raise RejectedSample(f"non_vehicle:{kind}")
    if not np.array_equal(target["timestep"].to_numpy(), np.arange(110)):
        raise RejectedSample("incomplete_or_duplicate_timesteps")
    observed = target["observed"].to_numpy(dtype=bool)
    if not (observed[:50].all() and not observed[50:].any()):
        raise RejectedSample("unexpected_observed")

    # Only this historical branch supplies model inputs and reference frame.
    past = target.iloc[:50]
    past_xy = past[["position_x", "position_y"]].to_numpy(float)  # [50, 2]
    past_vel = past[["velocity_x", "velocity_y"]].to_numpy(float)  # [50, 2]
    past_heading = past["heading"].to_numpy(float)  # [50]
    if not np.isfinite(np.r_[past_xy.ravel(), past_vel.ravel(), past_heading]).all():
        raise RejectedSample("nonfinite_history")
    
    # The last observed position and heading define the fixed local frame.
    origin, theta = past_xy[-1].copy(), float(past_heading[-1])
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    hist_xy = (past_xy - origin) @ rotation
    hist_vel = past_vel @ rotation
    relative_heading = past_heading - theta  # Relative historical heading.
    hist = np.column_stack([
        hist_xy / POS_SCALE, hist_vel / VEL_SCALE,
        np.sin(relative_heading), np.cos(relative_heading),
    ]).astype(np.float32)  # [50, 6]: position, velocity, and relative heading.

    # The future branch is a label only; never passed to the model encoder.
    future_xy = target.iloc[50:][["position_x", "position_y"]].to_numpy(float)
    if not np.isfinite(future_xy).all():
        raise RejectedSample("nonfinite_future")
    future_m = (future_xy - origin) @ rotation  # Future labels in local meters.
    times = np.arange(1, 61)[:, None] * DT
    cv_m = times * hist_vel[-1]  # [60, 2]: constant-velocity prediction in local meters.
  
    # Independent map-frame computation verifies frame-invariant errors.
    map_pred = origin + times * past_vel[-1]
    local_errors = np.linalg.norm(cv_m - future_m, axis=-1)
    map_errors = np.linalg.norm(map_pred - future_xy, axis=-1)
    np.testing.assert_allclose(local_errors, map_errors, atol=1e-8)
    np.testing.assert_allclose(hist_xy @ rotation.T + origin, past_xy, atol=1e-8)
    return {
        "scenario_id": str(df["scenario_id"].iloc[0]),
        "hist": hist,
        "hist_valid": np.ones(50, dtype=bool),
        "future_m": future_m.astype(np.float32),
        "future_valid": np.ones(60, dtype=bool),
        "cv_m": cv_m.astype(np.float32),
        "origin": origin, "theta": np.float64(theta),
    }


def read_sample(path):
    """Read a Parquet scenario and apply the declared sample protocol."""
    return encode_dataframe(pd.read_parquet(path))


def read_ids(path):
    ids = [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicated scenario list: {path}")
    if any("/" in s or "\\" in s or s in (".", "..") for s in ids):
        raise ValueError(f"Expected scenario IDs, not paths: {path}")
    return ids


def scenario_path(raw_root, split, scenario_id):
    return Path(raw_root) / split / scenario_id / f"scenario_{scenario_id}.parquet"


def save_cache(path, samples):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.stack([s[key] for s in samples]) for key in samples[0]}
    np.savez_compressed(path, **payload)


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_meta(cache_dir):
    """Validate cache version, coordinate scale, and file fingerprint."""
    directory = Path(cache_dir)
    meta = json.loads((directory / "metadata.json").read_text())
    if meta["version"] != VERSION or meta["position_scale_m"] != POS_SCALE:
        raise ValueError("Cache version or coordinate scale mismatch")
    actual = fingerprint([directory / "train.npz", directory / "val.npz"])
    if actual != meta["fingerprint"]:
        raise ValueError("Cache bytes changed; prepare a new cache version")
    return meta


def scene_metrics(pred_m, future_m):
    """Score one complete future in meters; equality at 2 m is not a miss."""
    error = np.linalg.norm(pred_m - future_m, axis=-1)
    if not np.isfinite(error).all():
        raise ValueError("Non-finite prediction error")
    return {
        "ADE": float(error.mean()),
        "FDE": float(error[-1]),
        "miss": int(error[-1] > 2.0),
    }


# ---------------------------------------------------------------------------
# 2. Frozen validation data and reproducible training subset selection
# ---------------------------------------------------------------------------


def load_validation_samples(raw_root, val_ids):
    """Load validation scenes in the frozen ID order; invalid scenes are errors."""
    samples = []
    for scenario_id in val_ids:
        path = scenario_path(raw_root, "val", scenario_id)
        sample = read_sample(path)
        if sample["scenario_id"] != scenario_id:
            raise ValueError(f"Filename/scenario_id mismatch: {scenario_id}")
        samples.append(sample)
    return samples


def verify_previous_cv(val_samples, val_ids, report_path):
    """Check scene IDs and CV scores against an optional previous baseline report."""
    if not report_path.exists():
        return

    previous_results = pd.read_csv(report_path).set_index("scenario_id")
    if set(previous_results.index) != set(val_ids) or not previous_results.index.is_unique:
        raise ValueError("Previous CV CSV and frozen validation IDs disagree")

    for sample in val_samples:
        scenario_id = sample["scenario_id"]
        current_score = scene_metrics(sample["cv_m"], sample["future_m"])
        for metric in ("ADE", "FDE"):
            # Allow at most 1 mm of numerical difference from the previous CV result.
            previous_score = float(previous_results.loc[scenario_id, metric])
            if abs(current_score[metric] - previous_score) > 1e-3:
                raise ValueError("CV mismatch (>1 mm); check units, velocity, and file version")
        if current_score["miss"] != int(previous_results.loc[scenario_id, "miss"]):
            raise ValueError("CV miss classification changed; inspect threshold precision")

    print("Local-coordinate CV agrees with previous map-coordinate CV.", flush=True)


def select_training_samples(raw_root, train_count, seed, val_ids):
    """Collect eligible training scenes and record selection/rejection reasons."""
    # Sort before seeded shuffling to reproduce selection for the same candidates.
    paths = sorted((raw_root / "train").glob("*/scenario_*.parquet"))
    random.Random(seed).shuffle(paths)

    samples = []
    scan_log = []
    validation_ids = set(val_ids)
    for path in paths:
        scenario_id = path.parent.name
        if scenario_id in validation_ids:
            raise ValueError("Train/validation scenario overlap")

        # read_sample checks vehicle type, complete timesteps, observed flags, and values.
        # Skip protocol rejections only; file corruption and other errors still propagate.
        try:
            sample = read_sample(path)
        except RejectedSample as exc:
            scan_log.append({"scenario_id": scenario_id, "status": str(exc)})
            continue

        if sample["scenario_id"] != scenario_id:
            raise ValueError(f"Filename/scenario_id mismatch: {path}")
        samples.append(sample)
        scan_log.append({"scenario_id": scenario_id, "status": "selected"})

        if len(samples) % 500 == 0:
            print(f"Training scenes: {len(samples)}/{train_count}", flush=True)
        if len(samples) == train_count:
            break

    if len(samples) != train_count:
        raise RuntimeError(f"Only found {len(samples)} eligible training scenes")
    train_ids = [sample["scenario_id"] for sample in samples]
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Duplicated training scenario ID")
    return samples, scan_log


def save_prepared_data(cache_dir, train_samples, val_samples, scan_log, *, seed, val_file):
    """Save arrays, scene IDs, selection logs, and metadata used by training."""
    cache_dir.mkdir(parents=True)

    # 1. Array caches avoid repeated Parquet preprocessing during training.
    save_cache(cache_dir / "train.npz", train_samples)
    save_cache(cache_dir / "val.npz", val_samples)

    # 2. Scene IDs and scan logs record exactly which data was used.
    train_ids = [sample["scenario_id"] for sample in train_samples]
    val_ids = [sample["scenario_id"] for sample in val_samples]
    (cache_dir / "train_ids.txt").write_text("\n".join(train_ids) + "\n")
    (cache_dir / "val_ids.txt").write_text("\n".join(val_ids) + "\n")
    pd.DataFrame(scan_log).to_csv(cache_dir / "train_scan.csv", index=False)

    # 3. Metadata records features, scales, CV scores, and cache fingerprints.
    scores = pd.DataFrame([
        scene_metrics(sample["cv_m"], sample["future_m"])
        for sample in val_samples
    ])
    metadata = {
        "version": VERSION,
        "position_scale_m": POS_SCALE,
        "velocity_scale_mps": VEL_SCALE,
        "dt_s": DT,
        "features": [
            "x/50", "y/50", "vx/20", "vy/20",
            "sin_relative_heading", "cos_relative_heading",
        ],
        "representation": "direct_local_position",
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "seed": seed,
        "val_source": str(val_file),
        "cv_summary": {
            metric: float(scores[metric].mean()) for metric in ("ADE", "FDE", "miss")
        },
        "fingerprint": fingerprint([cache_dir / "train.npz", cache_dir / "val.npz"]),
    }
    metadata_json = json.dumps(metadata, indent=2)
    (cache_dir / "metadata.json").write_text(metadata_json, encoding="utf-8")
    print(metadata_json, flush=True)
    print(f"Saved cache: {cache_dir}", flush=True)


def command_prepare(args):
    """Create a new cache while preserving the original train/validation split."""
    # Resolve paths and refuse to overwrite an existing cache.
    root = args.project_root.resolve()
    raw_root = root / "data/raw"
    cache_dir = root / "data/processed" / args.cache_name
    if cache_dir.exists():
        raise FileExistsError(f"Cache already exists: {cache_dir}; choose a new --cache-name")

    # Load the existing validation IDs without resampling.
    val_file = args.val_ids or root / "splits/val_cv100.txt"
    val_ids = read_ids(val_file)
    print(f"Reusing {len(val_ids)} fixed validation IDs from {val_file}", flush=True)
    val_samples = load_validation_samples(raw_root, val_ids)

    # Verify CV agreement when a previous baseline report is available.
    verify_previous_cv(val_samples, val_ids, root / "reports/cv_val100.csv")

    # Collect the requested number of eligible scenes from the training split.
    train_samples, scan_log = select_training_samples(
        raw_root, args.train_count, args.seed, val_ids
    )

    # Save prepared data to data/processed/<cache-name>/.
    save_prepared_data(
        cache_dir, train_samples, val_samples, scan_log,
        seed=args.seed, val_file=val_file,
    )


# ---------------------------------------------------------------------------
# 3. Cached Dataset, Transformer, and normalized position loss
# ---------------------------------------------------------------------------


class TrajectoryDataset(Dataset):
    """Load one cached split; return one normalized sample per index."""

    def __init__(self, cache_dir, split):
        with np.load(Path(cache_dir) / f"{split}.npz", allow_pickle=False) as archive:
            self.arrays = {key: archive[key] for key in archive.files}

    def __len__(self):
        return len(self.arrays["scenario_id"])

    def __getitem__(self, index):
        a = self.arrays
        return {
            "scenario_id": str(a["scenario_id"][index]),
            "hist": torch.from_numpy(a["hist"][index]),
            "hist_valid": torch.from_numpy(a["hist_valid"][index]),
            "future": torch.from_numpy(a["future_m"][index] / POS_SCALE),
            "future_valid": torch.from_numpy(a["future_valid"][index]),
            "cv_m": torch.from_numpy(a["cv_m"][index]),
        }


class HistoryTransformer(nn.Module):
    """Encode [B, 50, 6] histories and regress [B, 60, 2] future positions."""

    def __init__(self, d_model=128, heads=4, layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(6, d_model)
        self.time_embed = nn.Parameter(torch.randn(1, 50, d_model) * 0.02)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            block, num_layers=layers, enable_nested_tensor=False,
        )
        # Encoder clones would otherwise start with identical layer weights.
        for layer in self.encoder.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 60 * 2),
        )

    def forward(self, hist, hist_valid):
        tokens = self.input_proj(hist) + self.time_embed
        # Dataset: True means valid. PyTorch padding mask: True means ignore.
        memory = self.encoder(tokens, src_key_padding_mask=~hist_valid)
        # All selected focal histories include valid timestep 49.
        return self.head(memory[:, 49]).reshape(-1, 60, 2)


def position_loss(pred, future, future_valid):
    """Mean squared Euclidean distance over valid normalized future points."""
    squared_distance = (pred - future).square().sum(-1)
    return (squared_distance * future_valid).sum() / future_valid.sum().clamp_min(1)


# ---------------------------------------------------------------------------
# 4. Shared device, checkpoint, and evaluation utilities
# ---------------------------------------------------------------------------


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this interpreter; run check-environment first")
    return torch.device(name)


@torch.inference_mode()
def evaluate(model, loader, device):
    """Use the same per-scene meter-based metrics in training and evaluation."""
    model.eval()
    rows = []
    for batch in loader:
        pred_m = (
            model(batch["hist"].to(device), batch["hist_valid"].to(device)) * POS_SCALE
        ).cpu().numpy()
        future_m = batch["future"].numpy() * POS_SCALE
        for i, sid in enumerate(batch["scenario_id"]):
            result = scene_metrics(pred_m[i], future_m[i])
            cv = scene_metrics(batch["cv_m"][i].numpy(), future_m[i])
            rows.append({
                "scenario_id": sid,
                **result,
                "CV_ADE": cv["ADE"],
                "CV_FDE": cv["FDE"],
                "CV_miss": cv["miss"],
            })
    return pd.DataFrame(rows)


def summary(rows):
    return {
        key: float(rows[key].mean())
        for key in ("ADE", "FDE", "miss", "CV_ADE", "CV_FDE", "CV_miss")
    }


def load_model(checkpoint, device, cache_meta):
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if saved["cache_fingerprint"] != cache_meta["fingerprint"]:
        raise ValueError("Checkpoint and dataset cache differ")
    model = HistoryTransformer(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved


# ---------------------------------------------------------------------------
# 5. Training, evaluation, plotting, and diagnostic commands
# ---------------------------------------------------------------------------


def command_train(args):
    """Train in FP32, validate each epoch, and save best/last training states."""
    if args.epochs < 1 or args.batch_size < 1 or args.overfit < 0:
        raise ValueError("epochs/batch-size must be positive; overfit must be >=0")
    torch.set_num_threads(args.cpu_threads)
    # Load and verify the cache before starting or restoring an experiment.
    root = args.project_root.resolve()
    cache = root / "data/processed" / args.cache_name
    meta = load_meta(cache)
    device = resolve_device(args.device)
    seed_everything(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    config = {
        "d_model": args.d_model,
        "heads": args.heads,
        "layers": args.layers,
        "dropout": 0.0 if args.overfit else args.dropout,
    }
    saved = None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["cache_fingerprint"] != meta["fingerprint"]:
            raise ValueError("Resume cache mismatch")
        config = saved["model_config"]
        args.overfit = saved["overfit"]
        if args.batch_size != saved["batch_size"]:
            raise ValueError("Use the original batch size when resuming")
        run_dir = args.resume.resolve().parent
    else:
        run_dir = root / "checkpoints" / args.run
        if run_dir.exists():
            raise FileExistsError(f"Run exists: {run_dir}; use --resume or a new --run")
    # Use all cached training scenes, or a small subset for an overfit diagnostic.
    train_data = TrajectoryDataset(cache, "train")
    val_data = TrajectoryDataset(cache, "val")
    if args.overfit:
        if args.overfit > len(train_data):
            raise ValueError("overfit exceeds number of training samples")
        train_data = Subset(train_data, range(args.overfit))
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    replay_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    model = HistoryTransformer(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch, best, history = 0, float("inf"), []
    # Restore optimization and random states before continuing the training loop.
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if args.resume_lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = args.resume_lr
        start_epoch, best = saved["epoch"], saved["best_monitor_FDE"]
        if args.epochs <= start_epoch:
            raise ValueError("--epochs is the total target; it must exceed the saved epoch")
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        generator.set_state(saved["loader_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        log = run_dir / "training.csv"
        if log.exists():
            prior = pd.read_csv(log)
            history = prior[prior["epoch"] <= start_epoch].to_dict("records")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_record = {
        "model": config,
        "device": str(device),
        "batch_size": args.batch_size,
        "overfit": args.overfit,
        "cache": str(cache),
        "metadata": meta,
        "seed": args.seed,
        "workers": args.workers,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
        "torch_version": str(torch.__version__),
    }
    (run_dir / "config.json").write_text(json.dumps(config_record, indent=2))
    print(f"Device={device}; train={len(train_data)}, val={len(val_data)}", flush=True)
    print(f"Parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print("Monitor=" + ("TRAIN_REPLAY_FDE (diagnostic only)" if args.overfit else "VAL_DEV_FDE"), flush=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        weighted_loss, valid_points = 0.0, 0

        # Forward pass -> loss -> backward pass -> clipped optimizer update.
        for batch in train_loader:
            hist, mask = batch["hist"].to(device), batch["hist_valid"].to(device)
            future = batch["future"].to(device)
            future_mask = batch["future_valid"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(hist, mask)
            loss = position_loss(pred, future, future_mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss: inspect input and scale")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            points = int(future_mask.sum().item())
            weighted_loss += loss.item() * points
            valid_points += points
        # Normal runs monitor validation FDE; diagnostics monitor training replay.
        val = summary(evaluate(model, val_loader, device))
        replay = summary(evaluate(model, replay_loader, device)) if args.overfit else None
        monitor = replay["FDE"] if replay else val["FDE"]
        improved = monitor < best
        best = min(best, monitor)
        row = {
            "epoch": epoch,
            "train_loss_norm": weighted_loss / valid_points,
            "val_ADE_m": val["ADE"],
            "val_FDE_m": val["FDE"],
            "val_MR": val["miss"],
            "replay_ADE_m": replay["ADE"] if replay else None,
            "replay_FDE_m": replay["FDE"] if replay else None,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "training.csv", index=False)
        # Save enough state to resume; the cache fingerprint prevents data changes.
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": config,
            "epoch": epoch,
            "best_monitor_FDE": best,
            "overfit": args.overfit,
            "batch_size": args.batch_size,
            "cache_fingerprint": meta["fingerprint"],
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if improved:
            torch.save(checkpoint, run_dir / "best.pt")
        replay_note = f" | replayFDE={replay['FDE']:.3f}m" if replay else ""
        print(f"Epoch {epoch:03d} | loss={row['train_loss_norm']:.5f} | "
              f"valADE={val['ADE']:.3f}m | valFDE={val['FDE']:.3f}m | "
              f"MR={val['miss']:.1%}{replay_note}", flush=True)
    print(f"Saved: {run_dir}", flush=True)


def command_evaluate(args):
    """Compare a checkpoint and CV on the exact cached validation scenes."""
    root = args.project_root.resolve()
    cache = root / "data/processed" / args.cache_name
    meta = load_meta(cache)
    device = resolve_device(args.device)
    model, saved = load_model(args.checkpoint, device, meta)
    data = TrajectoryDataset(cache, "val")
    rows = evaluate(model, DataLoader(data, batch_size=args.batch_size, shuffle=False), device)
    if len(rows) != meta["val_count"]:
        raise RuntimeError("Validation scene count changed")
    result = {
        "N": len(rows),
        "K": 1,
        "unit": "m",
        "checkpoint_epoch": saved["epoch"],
        "cache_fingerprint": meta["fingerprint"],
        "overfit_diagnostic": bool(saved["overfit"]),
        **summary(rows),
    }
    run_name = args.checkpoint.resolve().parent.name
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows.to_csv(reports / f"transformer_{run_name}.csv", index=False)
    (reports / f"transformer_{run_name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if saved["overfit"]:
        print("Diagnostic checkpoint: this is not a full-data baseline.")
    print("Same validation scenes; all predictions scored in local meters.")


def command_visualize(args):
    """Plot a requested scene or the worst validation case after evaluation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.project_root.resolve()
    cache = root / "data/processed" / args.cache_name
    device = resolve_device(args.device)
    model, _ = load_model(args.checkpoint, device, load_meta(cache))
    run_name = args.checkpoint.resolve().parent.name
    sid = args.scenario_id
    if sid is None and args.split == "val":
        report_path = root / "reports" / f"transformer_{run_name}.csv"
        if not report_path.exists():
            raise FileNotFoundError(
                "Run evaluate first to select the worst case, or pass --scenario-id."
            )
        report = pd.read_csv(report_path)
        sid = str(report.sort_values("FDE", ascending=False).iloc[0]["scenario_id"])
        print("Post-hoc worst transformer FDE case:", sid)
    with np.load(cache / f"{args.split}.npz", allow_pickle=False) as a:
        if sid is None:
            sid = str(a["scenario_id"][0])
            print("First cached training scene:", sid)
        indices = np.flatnonzero(a["scenario_id"] == sid)
        if len(indices) != 1:
            raise ValueError("scenario-id must occur exactly once in the selected cached split")
        i = int(indices[0])
        hist = a["hist"][i].copy()
        valid = a["hist_valid"][i].copy()
        future, cv = a["future_m"][i].copy(), a["cv_m"][i].copy()
    with torch.inference_mode():
        pred = model(torch.from_numpy(hist[None]).to(device),
                     torch.from_numpy(valid[None]).to(device))[0].cpu().numpy() * POS_SCALE
    score = scene_metrics(pred, future)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    past = hist[:, :2] * POS_SCALE
    ax.plot(past[:, 0], past[:, 1], "k.-", label="History")
    ax.plot(future[:, 0], future[:, 1], "g.-", label="Ground truth")
    ax.plot(cv[:, 0], cv[:, 1], "r--", label="CV")
    ax.plot(pred[:, 0], pred[:, 1], color="royalblue", label="Transformer")
    ax.scatter(0, 0, c="black", s=35)
    ax.set(xlabel="Local forward x (m)", ylabel="Local left y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.3)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.03), ncol=2, fontsize=8)
    times = np.arange(1, 61) * .1
    axes[1].plot(times, np.linalg.norm(pred-future, axis=-1), label="Transformer")
    axes[1].plot(times, np.linalg.norm(cv-future, axis=-1), "r--", label="CV")
    axes[1].set(xlabel="Future time (s)", ylabel="Position error (m)",
                title=f"Transformer ADE={score['ADE']:.2f}m / FDE={score['FDE']:.2f}m")
    axes[1].grid(alpha=.3)
    axes[1].legend()
    fig.suptitle(sid, fontsize=10)
    fig.tight_layout()
    output = root / "figures" / run_name / f"{sid}.png"
    if args.split == "train":
        output = root / "figures" / run_name / "train" / f"{sid}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


def command_check_batch(args):
    """Check one batch, model outputs, finite loss, and backward gradients."""
    torch.set_num_threads(args.cpu_threads)
    print("Interpreter:", sys.executable)
    print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    device = resolve_device(args.device)
    cache = args.project_root.resolve() / "data/processed" / args.cache_name
    meta = load_meta(cache)
    print("Cache:", meta["train_count"], "train /", meta["val_count"], "val")
    data = TrajectoryDataset(cache, "train")
    batch = next(iter(DataLoader(data, batch_size=min(4, len(data)), shuffle=False)))
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(key, tuple(value.shape), value.dtype)
    print("First current normalized position:", batch["hist"][0, -1, :2].tolist())
    print("First current heading sin/cos:", batch["hist"][0, -1, 4:].tolist())
    model = HistoryTransformer().to(device)
    pred = model(batch["hist"].to(device), batch["hist_valid"].to(device))
    loss = position_loss(pred, batch["future"].to(device), batch["future_valid"].to(device))
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss during the batch check")
    loss.backward()
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError("Non-finite gradients during the batch check")
    print("Prediction:", tuple(pred.shape), "normalized positions")
    print("Finite loss:", bool(torch.isfinite(loss).item()))
    print("Forward/backward passed. This untrained loss is not a result.")


def command_check_environment(args):
    """Check dependency versions and run a real CUDA operation when available."""
    print("Python:", sys.version)
    print("Interpreter:", sys.executable)
    requirements = {
        "numpy": (1, 24), "pandas": (2, 0), "pyarrow": (12, 0),
        "matplotlib": (3, 7), "torch": (2, 2),
    }
    failures = []
    for name, minimum in requirements.items():
        try:
            module = importlib.import_module(name)
            version = module.__version__
            # These dependencies use numeric major/minor version components.
            installed = tuple(int(part) for part in version.split(".")[:2])
            print(f"{name}: {version} (required >= {minimum[0]}.{minimum[1]})")
            if installed < minimum:
                failures.append(f"{name} is below the required minimum version")
        except (ImportError, ValueError) as exc:
            failures.append(f"{name}: {exc}")

    print("Torch CUDA runtime:", torch.version.cuda)
    available = torch.cuda.is_available()
    print("CUDA available:", available)
    if available:
        print("GPU:", torch.cuda.get_device_name(0))
        values = torch.randn(256, 256, device="cuda")
        print("GPU matrix multiplication:", (values @ values.T).mean().item())
    elif args.require_cuda:
        failures.append("CUDA was requested but is unavailable in this interpreter")
    else:
        print("CPU execution is available; use --device cpu for model commands.")
    if failures:
        raise RuntimeError("Environment check failed: " + "; ".join(failures))
    print("Environment check passed.")


# ---------------------------------------------------------------------------
# 6. Public command-line interface
# ---------------------------------------------------------------------------


def build_parser():
    """Define a single public command-line interface for the full pipeline."""
    parser = argparse.ArgumentParser(
        description="History-only Transformer for single-agent vehicle trajectory prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Read the module docstring for requirements, data layout, and a full example workflow.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_command(name, help_text, handler, *, data=False, device=False):
        subparser = commands.add_parser(
            name, help=help_text, description=help_text,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        subparser.set_defaults(handler=handler)
        if data:
            subparser.add_argument(
                "--project-root", type=Path, default=Path.cwd(),
                help="Root containing data/, splits/, checkpoints/, reports/, and figures/",
            )
            subparser.add_argument(
                "--cache-name", default=VERSION,
                help="Cache directory name under data/processed/",
            )
        if device:
            subparser.add_argument(
                "--device", choices=["auto", "cuda", "cpu"], default="auto",
                help="Execution device; auto uses CUDA when available",
            )
        return subparser

    environment = add_command(
        "check-environment", "Check Python, dependencies, and CUDA execution", command_check_environment
    )
    environment.add_argument(
        "--require-cuda", action="store_true", help="Fail if CUDA is unavailable"
    )

    prepare = add_command(
        "prepare", "Build a training cache and reuse frozen validation IDs", command_prepare, data=True
    )
    prepare.add_argument("--train-count", type=int, default=5000, help="Number of eligible training scenes")
    prepare.add_argument("--seed", type=int, default=42, help="Seed for training scene selection")
    prepare.add_argument(
        "--val-ids", type=Path, default=None,
        help="Frozen validation ID file; relative paths are resolved under --project-root",
    )

    batch = add_command(
        "check-batch", "Check a cached batch with forward and backward passes",
        command_check_batch, data=True, device=True,
    )
    batch.add_argument("--cpu-threads", type=int, default=4, help="PyTorch CPU thread count")

    train = add_command(
        "train", "Train or resume the history-only Transformer", command_train, data=True, device=True
    )
    train.add_argument("--run", default="history_tf_v1", help="New experiment directory under checkpoints/")
    train.add_argument("--epochs", type=int, default=50, help="Total target epochs, including resumed epochs")
    train.add_argument("--batch-size", type=int, default=64, help="Scenes per batch; preserve this when resuming")
    train.add_argument("--lr", type=float, default=1e-3, help="Learning rate for a new run")
    train.add_argument("--seed", type=int, default=42, help="Random seed for a new run")
    train.add_argument(
        "--overfit", type=int, default=0,
        help="Diagnostic: use the first N training scenes and disable dropout; 0 uses all scenes",
    )
    train.add_argument("--d-model", type=int, default=128, help="Transformer hidden dimension")
    train.add_argument("--layers", type=int, default=3, help="Transformer encoder layer count")
    train.add_argument("--heads", type=int, default=4, help="Attention head count")
    train.add_argument("--dropout", type=float, default=0.1, help="Dropout probability for a normal run")
    train.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    train.add_argument("--cpu-threads", type=int, default=4, help="PyTorch CPU thread count")
    train.add_argument(
        "--resume", type=Path, default=None,
        help="Checkpoint to resume; relative paths are resolved under --project-root",
    )
    train.add_argument(
        "--resume-lr", type=float, default=None,
        help="Explicitly replace the saved optimizer learning rate when resuming",
    )

    evaluate_parser = add_command(
        "evaluate", "Compare Transformer and CV on the cached validation scenes",
        command_evaluate, data=True, device=True,
    )
    evaluate_parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Checkpoint path, relative to --project-root or absolute",
    )
    evaluate_parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size")
    evaluate_parser.add_argument("--cpu-threads", type=int, default=4, help="PyTorch CPU thread count")

    visualize = add_command(
        "visualize", "Plot predictions and errors for one cached scene",
        command_visualize, data=True, device=True,
    )
    visualize.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Checkpoint path, relative to --project-root or absolute",
    )
    visualize.add_argument(
        "--scenario-id", default=None,
        help="Cached scene ID; default: worst evaluated validation scene or first training scene",
    )
    visualize.add_argument("--split", choices=["train", "val"], default="val", help="Cached split to plot")
    visualize.add_argument("--cpu-threads", type=int, default=4, help="PyTorch CPU thread count")
    return parser


def main(argv=None):
    """Parse a subcommand, validate its options, and dispatch to its handler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("train_count", "epochs", "batch_size", "cpu_threads", "d_model", "heads", "layers"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("overfit", "workers"):
        if getattr(args, name, 0) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    for name in ("lr", "resume_lr"):
        value = getattr(args, name, None)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.command == "train":
        if not 0 <= args.dropout < 1:
            parser.error("--dropout must be in [0, 1)")
        if not args.resume and args.d_model % args.heads:
            parser.error("--d-model must be divisible by --heads")
        if args.resume_lr is not None and args.resume is None:
            parser.error("--resume-lr requires --resume")

    # Resolve user-supplied relative paths consistently against the project root.
    if hasattr(args, "project_root"):
        args.project_root = args.project_root.resolve()
        for name in ("val_ids", "resume", "checkpoint"):
            path = getattr(args, name, None)
            if path is not None and not path.is_absolute():
                setattr(args, name, args.project_root / path)
    if hasattr(args, "cpu_threads"):
        torch.set_num_threads(args.cpu_threads)
    args.handler(args)


if __name__ == "__main__":
    main()
