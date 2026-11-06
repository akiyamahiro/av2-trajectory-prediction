"""Plot a selected validation scene; optional worst-case selection is post-hoc analysis."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from data import POS_SCALE, load_meta, scene_metrics
from engine import resolve_device, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-name", default="history_v1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    root = args.project_root.resolve()
    cache = root / "data/processed" / args.cache_name
    device = resolve_device(args.device)
    model, _ = load_model(args.checkpoint, device, load_meta(cache))
    run_name = args.checkpoint.resolve().parent.name
    sid = args.scenario_id
    if sid is None and args.split == "val":
        report = pd.read_csv(root / "reports" / f"transformer_{run_name}.csv")
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
    axes[1].grid(alpha=.3); axes[1].legend()
    fig.suptitle(sid, fontsize=10); fig.tight_layout()
    output = root / "figures" / run_name / f"{sid}.png"
    if args.split == "train":
        output = root / "figures" / run_name / "train" / f"{sid}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
