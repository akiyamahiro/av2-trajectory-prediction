"""Pure NumPy/Pandas preprocessing. All model inputs use timesteps 0..49."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

VERSION = "history_v1"
POS_SCALE = 50.0
VEL_SCALE = 20.0
DT = 0.1


class RejectedSample(ValueError):
    """A valid file that does not meet our declared vehicle-only protocol."""


def encode_dataframe(df):
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
    past_xy = past[["position_x", "position_y"]].to_numpy(float) #(50, 2)
    past_vel = past[["velocity_x", "velocity_y"]].to_numpy(float) #(50, 2)
    past_heading = past["heading"].to_numpy(float) #(50,)
    if not np.isfinite(np.r_[past_xy.ravel(), past_vel.ravel(), past_heading]).all():
        raise RejectedSample("nonfinite_history")
    
    origin, theta = past_xy[-1].copy(), float(past_heading[-1]) #last historical position and heading as current state
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    hist_xy = (past_xy - origin) @ rotation
    hist_vel = past_vel @ rotation
    relative_heading = past_heading - theta #相对朝向
    hist = np.column_stack([
        hist_xy / POS_SCALE, hist_vel / VEL_SCALE,
        np.sin(relative_heading), np.cos(relative_heading),
    ]).astype(np.float32) #（50, 6）历史状态特征：位置、速度、相对朝向的正弦和余弦

    # The future branch is a label only; never passed to the model encoder.
    future_xy = target.iloc[50:][["position_x", "position_y"]].to_numpy(float)
    if not np.isfinite(future_xy).all():
        raise RejectedSample("nonfinite_future")
    future_m = (future_xy - origin) @ rotation #future positions in local coordinates
    times = np.arange(1, 61)[:, None] * DT
    cv_m = times * hist_vel[-1] #（60, 2）future positions predicted by constant velocity model in local coordinates
  
    # Independent map-frame computation verifies frame-invariant errors.
    map_pred = origin + times * past_vel[-1] #future positions predicted by constant velocity model in map coordinates
    local_errors = np.linalg.norm(cv_m - future_m, axis=-1)
    map_errors = np.linalg.norm(map_pred - future_xy, axis=-1)
    np.testing.assert_allclose(local_errors, map_errors, atol=1e-8)
    np.testing.assert_allclose(hist_xy @ rotation.T + origin, past_xy, atol=1e-8)
    return {
        "scenario_id": str(df["scenario_id"].iloc[0]),
        "hist": hist, "hist_valid": np.ones(50, dtype=bool),
        "future_m": future_m.astype(np.float32),
        "future_valid": np.ones(60, dtype=bool),
        "cv_m": cv_m.astype(np.float32),
        "origin": origin, "theta": np.float64(theta),
    }


def read_sample(path):
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
    directory = Path(cache_dir)
    meta = json.loads((directory / "metadata.json").read_text())
    if meta["version"] != VERSION or meta["position_scale_m"] != POS_SCALE:
        raise ValueError("Cache version or coordinate scale mismatch")
    actual = fingerprint([directory / "train.npz", directory / "val.npz"])
    if actual != meta["fingerprint"]:
        raise ValueError("Cache bytes changed; prepare a new cache version")
    return meta


def scene_metrics(pred_m, future_m):
    error = np.linalg.norm(pred_m - future_m, axis=-1)
    if not np.isfinite(error).all():
        raise ValueError("Non-finite prediction error")
    return {"ADE": float(error.mean()), "FDE": float(error[-1]),
            "miss": int(error[-1] > 2.0)}
