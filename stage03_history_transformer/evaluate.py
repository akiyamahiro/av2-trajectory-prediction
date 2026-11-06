"""K=1 comparison on the exact validation IDs stored by prepare.py."""
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
from data import load_meta
from dataset import TrajectoryDataset
from engine import resolve_device, load_model, evaluate, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-name", default="history_v1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    root = args.project_root.resolve()
    cache = root / "data/processed" / args.cache_name
    meta = load_meta(cache)
    device = resolve_device(args.device)
    model, saved = load_model(args.checkpoint, device, meta)
    data = TrajectoryDataset(cache, "val")
    rows = evaluate(model, DataLoader(data, batch_size=args.batch_size, shuffle=False), device)
    if len(rows) != meta["val_count"]:
        raise RuntimeError("Validation scene count changed")
    result = {"N": len(rows), "K": 1, "unit": "m", "checkpoint_epoch": saved["epoch"],
              "cache_fingerprint": meta["fingerprint"], "overfit_diagnostic": bool(saved["overfit"]),
              **summary(rows)}
    run_name = args.checkpoint.resolve().parent.name
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows.to_csv(reports / f"transformer_{run_name}.csv", index=False)
    (reports / f"transformer_{run_name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if saved["overfit"]:
        print("Diagnostic checkpoint: this is not a full-data baseline.")
    print("Same validation scenes; all predictions scored in local meters.")


if __name__ == "__main__":
    main()
