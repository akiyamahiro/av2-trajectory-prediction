"""Shared evaluation ensures training and standalone evaluation use one protocol."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from data import POS_SCALE, scene_metrics
from model import HistoryTransformer


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
        raise RuntimeError("CUDA unavailable in this interpreter; run inspect.py first")
    return torch.device(name)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        pred_m = (model(batch["hist"].to(device), batch["hist_valid"].to(device))
                  * POS_SCALE).cpu().numpy()
        future_m = batch["future"].numpy() * POS_SCALE
        for i, sid in enumerate(batch["scenario_id"]):
            result = scene_metrics(pred_m[i], future_m[i])
            cv = scene_metrics(batch["cv_m"][i].numpy(), future_m[i])
            rows.append({"scenario_id": sid, **result,
                         "CV_ADE": cv["ADE"], "CV_FDE": cv["FDE"], "CV_miss": cv["miss"]})
    return pd.DataFrame(rows)


def summary(rows):
    return {key: float(rows[key].mean()) for key in
            ("ADE", "FDE", "miss", "CV_ADE", "CV_FDE", "CV_miss")}


def load_model(checkpoint, device, cache_meta):
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if saved["cache_fingerprint"] != cache_meta["fingerprint"]:
        raise ValueError("Checkpoint and dataset cache differ")
    model = HistoryTransformer(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved
