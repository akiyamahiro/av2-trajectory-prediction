"""FP32 first: train replay diagnostic, fixed validation, best/last checkpoints."""
import argparse
from pathlib import Path
import json
import time
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from data import load_meta
from dataset import TrajectoryDataset
from model import HistoryTransformer, position_loss
from engine import seed_everything, resolve_device, evaluate, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-name", default="history_v1")
    parser.add_argument("--run", default="history_5000_lre-3")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=300, help="Total epochs, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit", type=int, default=0, help="Training replay diagnostic on first N training samples")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-lr", type=float, default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.overfit < 0:
        parser.error("epochs/batch-size must be positive; overfit must be >=0")
    torch.set_num_threads(args.cpu_threads)
    root = args.project_root.resolve()
    cache = root / "data/processed" / args.cache_name
    meta = load_meta(cache)
    device = resolve_device(args.device)
    seed_everything(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    config = {"d_model": args.d_model, "heads": args.heads,
              "layers": args.layers, "dropout": 0.0 if args.overfit else args.dropout}
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
    train_data = TrajectoryDataset(cache, "train")
    val_data = TrajectoryDataset(cache, "val")
    if args.overfit:
        if args.overfit > len(train_data):
            raise ValueError("overfit exceeds number of training samples")
        train_data = Subset(train_data, range(args.overfit))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=generator, num_workers=args.workers,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)
    replay_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    model = HistoryTransformer(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch, best, history = 0, float("inf"), []
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
    config_record = {"model": config, "device": str(device), "batch_size": args.batch_size,
                     "overfit": args.overfit, "cache": str(cache), "metadata": meta,
                     "seed": args.seed, "workers": args.workers,
                     "learning_rate": optimizer.param_groups[0]["lr"],
                     "weight_decay": optimizer.param_groups[0]["weight_decay"],
                     "torch_version": str(torch.__version__)}
    (run_dir / "config.json").write_text(json.dumps(config_record, indent=2))
    print(f"Device={device}; train={len(train_data)}, val={len(val_data)}", flush=True)
    print(f"Parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print("Monitor=" + ("TRAIN_REPLAY_FDE (diagnostic only)" if args.overfit else "VAL_DEV_FDE"), flush=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        weighted_loss, valid_points = 0.0, 0
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
        val = summary(evaluate(model, val_loader, device))
        replay = summary(evaluate(model, replay_loader, device)) if args.overfit else None
        monitor = replay["FDE"] if replay else val["FDE"]
        improved = monitor < best
        best = min(best, monitor)
        row = {"epoch": epoch, "train_loss_norm": weighted_loss / valid_points,
               "val_ADE_m": val["ADE"], "val_FDE_m": val["FDE"], "val_MR": val["miss"],
               "replay_ADE_m": replay["ADE"] if replay else None,
               "replay_FDE_m": replay["FDE"] if replay else None,
               "seconds": time.perf_counter() - started}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "training.csv", index=False)
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "model_config": config, "epoch": epoch, "best_monitor_FDE": best,
            "overfit": args.overfit, "batch_size": args.batch_size,
            "cache_fingerprint": meta["fingerprint"],
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
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


if __name__ == "__main__":
    main()
