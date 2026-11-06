"""准备 Transformer 使用的数据缓存。

阅读时可以先看文件末尾的 main()，再按调用顺序查看各个函数：
读取参数 → 读取固定验证集 → 核对旧 CV 结果 → 筛选训练集 → 保存缓存。

本文件负责组织数据准备流程；单个场景的筛选、坐标变换和特征构造
由 data.py 中的 read_sample() / encode_dataframe() 完成。
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from data import (
    DT,
    POS_SCALE,
    VEL_SCALE,
    VERSION,
    RejectedSample,
    fingerprint,
    read_ids,
    read_sample,
    save_cache,
    scenario_path,
    scene_metrics,
)


def parse_args():
    """读取命令行参数；未指定的选项使用默认值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd(), help="项目根目录"
    )
    parser.add_argument(
        "--train-count", type=int, default=5000, help="需要收集的合格训练场景数"
    )
    parser.add_argument("--seed", type=int, default=42, help="训练场景打乱顺序的随机种子")
    parser.add_argument(
        "--val-ids", type=Path, default=None,
        help="固定验证场景清单，默认使用 splits/val_cv100.txt",
    )
    parser.add_argument(
        "--cache-name", default=VERSION, help="data/processed 下的缓存目录名"
    )
    args = parser.parse_args()
    if args.train_count < 1:
        parser.error("train-count must be positive")
    return args


def load_validation_samples(raw_root, val_ids):
    """按已有 ID 清单读取验证场景，保持顺序；缺失或不合格时直接报错。"""
    samples = []
    for scenario_id in val_ids:
        path = scenario_path(raw_root, "val", scenario_id)
        sample = read_sample(path)
        if sample["scenario_id"] != scenario_id:
            raise ValueError(f"Filename/scenario_id mismatch: {scenario_id}")
        samples.append(sample)
    return samples


def verify_previous_cv(val_samples, val_ids, report_path):
    """若旧 CV 报告存在，核对场景清单、ADE/FDE 和 miss，保证比较口径一致。"""
    if not report_path.exists():
        return

    previous_results = pd.read_csv(report_path).set_index("scenario_id")
    if set(previous_results.index) != set(val_ids) or not previous_results.index.is_unique:
        raise ValueError("Previous CV CSV and frozen validation IDs disagree")

    for sample in val_samples:
        scenario_id = sample["scenario_id"]
        current_score = scene_metrics(sample["cv_m"], sample["future_m"])
        for metric in ("ADE", "FDE"):
            # 新旧 CV 的距离误差允许最多 1 mm 的数值差异。
            previous_score = float(previous_results.loc[scenario_id, metric])
            if abs(current_score[metric] - previous_score) > 1e-3:
                raise ValueError("CV mismatch (>1 mm); check units, velocity, and file version")
        if current_score["miss"] != int(previous_results.loc[scenario_id, "miss"]):
            raise ValueError("CV miss classification changed; inspect threshold precision")

    print("Local-coordinate CV agrees with previous map-coordinate CV.", flush=True)


def select_training_samples(raw_root, train_count, seed, val_ids):
    """从原始 train 目录收集合格场景，同时记录入选和排除原因。"""
    # 先排序，再固定种子打乱；在候选数据相同的情况下，筛选顺序可复现。
    paths = sorted((raw_root / "train").glob("*/scenario_*.parquet"))
    random.Random(seed).shuffle(paths)

    samples = []
    scan_log = []
    validation_ids = set(val_ids)
    for path in paths:
        scenario_id = path.parent.name
        if scenario_id in validation_ids:
            raise ValueError("Train/validation scenario overlap")

        # data.py 检查目标是否为 vehicle、110 帧是否齐全、observed 划分及数值等。
        # 仅跳过明确不符合样本协议的场景；文件损坏等其他异常继续向外报错。
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
    """保存数组缓存、场景清单、筛选记录，以及供训练程序核对的元数据。"""
    cache_dir.mkdir(parents=True)

    # 1. 数组缓存：训练时直接读取，无需每轮重新处理 Parquet。
    save_cache(cache_dir / "train.npz", train_samples)
    save_cache(cache_dir / "val.npz", val_samples)

    # 2. 场景清单与筛选记录：用于追溯本次使用了哪些数据。
    train_ids = [sample["scenario_id"] for sample in train_samples]
    val_ids = [sample["scenario_id"] for sample in val_samples]
    (cache_dir / "train_ids.txt").write_text("\n".join(train_ids) + "\n")
    (cache_dir / "val_ids.txt").write_text("\n".join(val_ids) + "\n")
    pd.DataFrame(scan_log).to_csv(cache_dir / "train_scan.csv", index=False)

    # 3. 元数据：保存特征含义、归一化尺度、CV 基线和缓存文件校验值。
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


def main():
    """数据准备入口：按顺序组织各步骤。"""
    # 第一步：读取参数，确定输入、输出路径，并避免覆盖已有缓存。
    args = parse_args()
    root = args.project_root.resolve()
    raw_root = root / "data/raw"
    cache_dir = root / "data/processed" / args.cache_name
    if cache_dir.exists():
        raise FileExistsError(f"Cache already exists: {cache_dir}; choose a new --cache-name")

    # 第二步：复用之前固定的验证清单，不重新随机抽取验证场景。
    val_file = args.val_ids or root / "splits/val_cv100.txt"
    val_ids = read_ids(val_file)
    print(f"Reusing {len(val_ids)} fixed validation IDs from {val_file}", flush=True)
    val_samples = load_validation_samples(raw_root, val_ids)

    # 第三步：如果已有 CV 报告，确认当前数据处理得到相同的基线结果。
    verify_previous_cv(val_samples, val_ids, root / "reports/cv_val100.csv")

    # 第四步：从原始训练集中，按固定随机顺序收集指定数量的合格样本。
    train_samples, scan_log = select_training_samples(
        raw_root, args.train_count, args.seed, val_ids
    )

    # 第五步：将准备好的数据保存到 data/processed/<cache-name>/。
    save_prepared_data(
        cache_dir, train_samples, val_samples, scan_log,
        seed=args.seed, val_file=val_file,
    )


if __name__ == "__main__":
    main()
