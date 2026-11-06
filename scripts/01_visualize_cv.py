from pathlib import Path # 用来表示和操作文件路径
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser() #创建一个参数解析器对象，并赋值给变量parser
    parser.add_argument("--parquet",type=Path,required=True) 
    parser.add_argument("--output",type=Path,default=Path("figures/cv_one.png"))
    args = parser.parse_args()

    #指定指定预测对象，按时间排序
    df = pd.read_parquet(args.parquet) #读取并加载parquet文件到DataFrame表格中
    focal_id = df['focal_track_id'].iloc[0] #取出focal_track_id列， 并取其中第一个id对象
    target = df.loc[df["track_id"] == focal_id].copy() #取出所有track_id等于focal_id（目标）的行，并创建一个副本
    target = target.sort_values("timestep") #按照时间戳升序排序

    object_type = str(target["object_type"].iloc[0]).upper()
    print(f"Focal agent type: {object_type}")
    # if str(target["object_type"].iloc[0]).lower() != "vehicle":
    #     raise ValueError("This script expects a vehicle focal agent")

    ts = target["timestep"].to_numpy()
    observed = target["observed"].to_numpy(dtype=bool)

    if not np.array_equal(ts,np.arange(110)):
        raise ValueError("Expected unique complete timesteps 0...109")
    if not(observed[:50].all() and not observed[50:].any()):
        raise ValueError("Unexpected history/future split")
    
    xy = target[["position_x", "position_y"]].to_numpy(dtype=float)
    history_map = xy[:50]  # [50, 2]
    future_map = xy[50:]   # [60, 2]，只用于评测和绘图 

    #2 仅用最后一个历史状态定义当前状态
    origin = history_map[-1].copy()  #坐标 (2,)，当前车辆位置
    theta = float(target["heading"].iloc[49]) #朝向角
    velocity_map = target[["velocity_x","velocity_y"]].iloc[49].to_numpy(dtype=float) #速度 shape (2,)

    if not np.isfinite(np.r_[xy.ravel(),velocity_map,theta]).all():
        raise ValueError("Non-finite position, velocity, or heading")

    #3 地图坐标 -> 目标车辆坐标
    c,s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c,-s],[s,c]])

    history = (history_map - origin) @ rotation #shape[50,2]
    future = (future_map - origin) @ rotation
    velocity = velocity_map @ rotation #速度只旋转不平移 #[1,2]

    #检查: 当前点应该为原点；逆变换应恢复原始历史坐标
    np.testing.assert_allclose(history[-1],0.0,atol=1e-8)
    np.testing.assert_allclose(history@rotation.T + origin, history_map,atol=1e-8)


    # 4. CV预测：未来第1点在0.1秒，第60点在6.0秒
    dt = 0.1
    future_times = np.arange(1, 61) * dt #[0.1, 0.2, ..., 6.0] shape(60,)
    prediction = history[-1] + future_times[:, None] * velocity #shape(2,) + shape(60,1)* (2,)

    # 5. 单轨迹误差，单位：米
    errors = np.linalg.norm(prediction - future, axis=-1) #shape(60,2) -> shape(60,)
    ade = errors.mean()
    fde = errors[-1]
    miss = int(fde > 2.0)

    scenario_id = str(df["scenario_id"].iloc[0])
    print("Scenario:", scenario_id)
    print("History / future / prediction:",
          history.shape, future.shape, prediction.shape)
    print(f"Current speed: {np.linalg.norm(velocity):.3f} m/s")
    print("Local velocity [forward, left]:", velocity)
    print(f"ADE: {ade:.3f} m")
    print(f"FDE: {fde:.3f} m")
    print(f"Miss (> 2 m): {miss}")

    # 6. 一张轨迹图 + 一张随时间变化的误差图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(history[:, 0], history[:, 1], "k.-", label="History")
    ax.plot(future[:, 0], future[:, 1], "g.-", label="Ground truth")
    ax.plot(prediction[:, 0], prediction[:, 1], "r--", label="CV")
    ax.scatter(0, 0, c="blue", s=50, label="Current", zorder=5)
    ax.set_xlabel("Local x: forward (m)")
    ax.set_ylabel("Local y: left (m)")
    ax.set_title("Vehicle trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(future_times, errors)
    ax.axhline(2.0, color="gray", linestyle="--", label="2 m reference")
    ax.set_xlabel("Time into future (s)")
    ax.set_ylabel("Position error (m)")
    ax.set_title(f"ADE={ade:.2f} m | FDE={fde:.2f} m | Miss={miss}")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(scenario_id, fontsize=10)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print("Saved:", args.output.resolve())


if __name__ == "__main__":
    main()

"""
#Automatically visualize the CV prediction best and worst:
from pathlib import Path
import subprocess
import sys

import pandas as pd

root = Path.cwd()
results = pd.read_csv(root / "reports/cv_val100.csv")
ordered = results.sort_values("FDE")

groups = {
    "best": ordered.head(2),
    "worst": ordered.tail(2),
}

for group, rows in groups.items():
    for _, row in rows.iterrows():
        scenario_id = row["scenario_id"]

        parquet = (
            root / "data/raw/val" / scenario_id
            / f"scenario_{scenario_id}.parquet"
        )
        output = (
            root / "figures/cv_cases"
            / f"{group}_{scenario_id}.png"
        )

        print(
            f"{group}: {scenario_id} | "
            f"ADE={row['ADE']:.3f}, FDE={row['FDE']:.3f}",
            flush=True,
        )

        subprocess.run(
            [
                sys.executable,
                str(root / "scripts/01_visualize_cv.py"),
                "--parquet", str(parquet),
                "--output", str(output),
            ],
            check=True,
        )
        """