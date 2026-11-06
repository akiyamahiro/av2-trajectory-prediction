"""Check the selected interpreter, GPU, cache, Dataset, batch, and model shapes."""
import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
from data import load_meta
from dataset import TrajectoryDataset
from model import HistoryTransformer, position_loss
from engine import resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-name", default="history_v1")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    print("Interpreter:", sys.executable)
    print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    device = resolve_device(args.device)
    cache = args.project_root.resolve()/ "data/processed" / args.cache_name
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
    loss.backward()
    print("Prediction:", tuple(pred.shape), "normalized positions")
    print("Finite loss:", bool(torch.isfinite(loss).item()))
    print("Forward/backward passed. This untrained loss is not a result.")


if __name__ == "__main__":
    main()
