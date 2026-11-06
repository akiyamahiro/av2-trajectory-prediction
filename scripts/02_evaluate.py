from pathlib import Path
import random

import numpy as np
import pandas as pd


def load_vehicle_target(path):
    """读取目标轨迹；不符合当前任务范围时返回原因。任务范围：种类，时间戳范围，历史/未来分割"""
    df = pd.read_parquet(path)
    focal_id = df["focal_track_id"].iloc[0]
    target = df.loc[df["track_id"] == focal_id].copy()
    target = target.sort_values("timestep")

    kind = str(target["object_type"].iloc[0]).lower()
    if kind != "vehicle":
        return None, f"Non-vehicle：{kind}"
    
    ts = target["timestep"].to_numpy()
    if not np.array_equal(ts, np.arange(110)):
        return None, "Incomplete timesteps"

    observed = target["observed"].to_numpy(dtype=bool)
    if not np.all(observed[:50]) or np.any(observed[50:]):
        return None, "Unexpected observed timesteps"

    return target, "selected"



def main():
    data_root = Path.home() / "YANG/00trajdff/data/raw"
    report_dir = Path("reports")
    split_dir = Path("splits")

    num_scenarios = 100
    seed = 42
    dt = 0.1

    # 排序后再固定种子打乱，使选择过程可复现。
    paths = sorted(
        (data_root / "val").glob("*/scenario_*.parquet")
    )
    if not paths:
        raise FileNotFoundError("没有找到Validation场景，请检查路径。")

    random.Random(seed).shuffle(paths) #

    results = []
    scan_log = []

    for path in paths:
        target, status = load_vehicle_target(path)

        scan_log.append({
            "scenario_id": path.parent.name,
            "status": status,
        })

        if target is None:
            continue

        xy = target[
            ["position_x", "position_y"]
        ].to_numpy(dtype=float)

        velocity = target[
            ["velocity_x", "velocity_y"]
        ].iloc[49].to_numpy(dtype=float)

        if not np.isfinite(np.r_[xy.ravel(), velocity]).all():
            raise ValueError(f"存在非有限数值：{path}")

        # 在地图坐标系直接计算CV，单位仍是米。
        current_position = xy[49]
        future = xy[50:]
        future_times = np.arange(1, 61) * dt

        prediction = (
            current_position
            + future_times[:, None] * velocity
        )

        errors = np.linalg.norm(prediction - future, axis=-1)

        results.append({
            "scenario_id": path.parent.name,
            "ADE": float(errors.mean()),
            "FDE": float(errors[-1]),
            "miss": int(errors[-1] > 2.0),
        })

        if len(results) % 20 == 0:
            print(f"已评测 {len(results)} / {num_scenarios}")

        if len(results) == num_scenarios:
            break

    if len(results) != num_scenarios:
        raise RuntimeError(
            f"合格场景不足：只找到 {len(results)} 个。"
        )

    report_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(results)
    result_df.to_csv(report_dir / "cv_val100.csv", index=False)

    pd.DataFrame(scan_log).to_csv(
        report_dir / "cv_val100_scan.csv", index=False
    )

    # 固定场景清单：以后所有模型使用相同场景比较。
    selected_ids = result_df["scenario_id"].tolist()
    (split_dir / "val_cv100.txt").write_text(
        "\n".join(selected_ids) + "\n",
        encoding="utf-8",
    )

    print("\n=== CV Validation Baseline ===")
    print(f"有效场景数：{len(result_df)}")
    print(f"平均 ADE：{result_df['ADE'].mean():.3f} m")
    print(f"平均 FDE：{result_df['FDE'].mean():.3f} m")
    print(f"Miss Rate：{result_df['miss'].mean():.1%}")
    print(f"扫描场景数：{len(scan_log)}")
    print("\n筛选情况：")
    print(pd.DataFrame(scan_log)["status"].value_counts())


if __name__ == "__main__":
    main()