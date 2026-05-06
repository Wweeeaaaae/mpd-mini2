# 脚本功能：按 seed 和指定 joint 轨迹，反推出固定速度动态球的初始三维位置。
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pinocchio as pin

try:
    from .inference_trajectory_loader import load_inference_reference
except ImportError:
    from inference_trajectory_loader import load_inference_reference
try:
    from .asset_paths import mini2_urdf_path
except ImportError:
    from asset_paths import mini2_urdf_path


# 动态球速度保持固定，单位 m/s；脚本只反推动态球初始位置，不改变速度。
FIXED_DYNAMIC_SPHERE_V = np.array([0.06, -0.07, 0.03], dtype=float)
# 动态球半径保持固定，单位 m；需与 cascaded_mpc.py 中动态球半径一致。
FIXED_DYNAMIC_SPHERE_RADIUS = 0.06
# inference 参考轨迹默认总时长，单位 s；用于把 hit_step 换算成 hit_time。
DEFAULT_TRAJECTORY_DURATION = 5.0
# 动态球命中参考轨迹的归一化时刻；0.4 表示在整条轨迹 40% 附近制造影响。
DEFAULT_HIT_FRACTION = 0.4
# 默认分别针对机械臂 4/5/6 号关节附近生成动态球初始位置候选。
DEFAULT_HIT_FRAMES = ("4_joint", "5_joint", "6_joint")
# 碰撞强度，范围 [0, 1]；1 表示球心穿过 joint，0 表示只擦到安全包络边界。
DEFAULT_COLLISION_STRENGTH = 0.8
# 安全裕度，单位 m；与动态球半径相加后形成用于定义碰撞强度的有效半径。
DEFAULT_SAFETY_MARGIN = 0.04


def _compute_frame_positions(q_ref: np.ndarray, frame_names: Sequence[str]):
    model = pin.buildModelFromUrdf(str(mini2_urdf_path()))
    data = model.createData()
    frame_ids = {}
    for frame_name in frame_names:
        if not model.existFrame(frame_name):
            raise RuntimeError(f"找不到 frame：{frame_name}")
        frame_ids[frame_name] = model.getFrameId(frame_name)

    positions = {frame_name: [] for frame_name in frame_names}
    for q in q_ref:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for frame_name, frame_id in frame_ids.items():
            p = np.asarray(data.oMf[frame_id].translation).reshape(3).copy()
            positions[frame_name].append(p)
    return {
        frame_name: np.asarray(frame_positions, dtype=float)
        for frame_name, frame_positions in positions.items()
    }


def _collision_offset_direction(frame_positions: np.ndarray, hit_step: int):
    if frame_positions.shape[0] <= 1:
        return np.array([1.0, 0.0, 0.0], dtype=float)

    prev_step = max(hit_step - 1, 0)
    next_step = min(hit_step + 1, frame_positions.shape[0] - 1)
    tangent = frame_positions[next_step] - frame_positions[prev_step]
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-9:
        tangent = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        tangent = tangent / tangent_norm

    direction = np.cross(tangent, np.array([0.0, 0.0, 1.0], dtype=float))
    if np.linalg.norm(direction) < 1e-9:
        direction = np.cross(tangent, np.array([0.0, 1.0, 0.0], dtype=float))
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return direction / direction_norm


def compute_dynamic_sphere_position(
    seed: int,
    hit_fraction: float = DEFAULT_HIT_FRACTION,
    trajectory_duration: float = DEFAULT_TRAJECTORY_DURATION,
    results_root: Optional[Path] = None,
    hit_frames: Sequence[str] = DEFAULT_HIT_FRAMES,
    collision_strength: float = DEFAULT_COLLISION_STRENGTH,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    dynamic_sphere_velocity: np.ndarray = FIXED_DYNAMIC_SPHERE_V,
    dynamic_sphere_radius: float = FIXED_DYNAMIC_SPHERE_RADIUS,
):
    """返回每个指定 joint 对应的动态球初始三维位置。"""
    if not 0.0 <= hit_fraction <= 1.0:
        raise ValueError("hit_fraction 必须位于 [0, 1] 区间内")
    if not 0.0 <= collision_strength <= 1.0:
        raise ValueError("collision_strength 必须位于 [0, 1] 区间内")
    if safety_margin < 0.0:
        raise ValueError("safety_margin 不能为负数")
    if not hit_frames:
        raise ValueError("hit_frames 至少需要包含一个 frame 名称")
    dynamic_sphere_velocity = np.asarray(dynamic_sphere_velocity, dtype=float).reshape(-1)
    if dynamic_sphere_velocity.shape != (3,):
        raise ValueError("dynamic_sphere_velocity 必须是 3 维向量")
    dynamic_sphere_radius = float(dynamic_sphere_radius)
    if dynamic_sphere_radius <= 0.0:
        raise ValueError("dynamic_sphere_radius 必须为正数")

    inference_ref = load_inference_reference(
        seed=seed,
        results_root=results_root,
        end_frame_name=hit_frames[-1],
    )
    frame_positions = _compute_frame_positions(inference_ref.q_ref, hit_frames)
    n_steps = inference_ref.q_ref.shape[0]
    hit_step = int(round((n_steps - 1) * hit_fraction))
    dt = trajectory_duration / n_steps
    hit_time = hit_step * dt
    effective_radius = dynamic_sphere_radius + safety_margin
    offset_distance = (1.0 - collision_strength) * effective_radius

    by_frame = {}
    for frame_name in hit_frames:
        positions = frame_positions[frame_name]
        target_position = positions[hit_step].copy()
        offset_direction = _collision_offset_direction(positions, hit_step)
        sphere_position_at_hit = target_position + offset_direction * offset_distance
        dynamic_sphere_c0 = sphere_position_at_hit - dynamic_sphere_velocity * hit_time
        by_frame[frame_name] = {
            "target_position": target_position,
            "offset_direction": offset_direction,
            "sphere_position_at_hit": sphere_position_at_hit,
            "dynamic_sphere_position": dynamic_sphere_c0,
            "distance_to_joint_at_hit": float(offset_distance),
            "penetration_depth_vs_effective_radius": float(effective_radius - offset_distance),
        }

    return {
        "seed": seed,
        "idx_best_traj": inference_ref.idx_best_traj,
        "hit_fraction": float(hit_fraction),
        "hit_step": int(hit_step),
        "hit_time": float(hit_time),
        "dynamic_sphere_velocity": dynamic_sphere_velocity.copy(),
        "dynamic_sphere_radius": float(dynamic_sphere_radius),
        "safety_margin": float(safety_margin),
        "effective_radius": float(effective_radius),
        "collision_strength": float(collision_strength),
        "by_frame": by_frame,
    }


def _format_array(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.8f}" for v in values) + "]"


def main():
    parser = argparse.ArgumentParser(
        description="根据指定 seed 和 joint 轨迹生成固定速度动态球的初始三维位置。"
    )
    parser.add_argument("--seed", type=int, default=4, help="读取 scripts/inference/logs/mini2/<seed> 的结果。")
    parser.add_argument("--hit-fraction", type=float, default=DEFAULT_HIT_FRACTION,
                        help="动态球命中参考轨迹的归一化时刻，0 表示起点，1 表示终点。")
    parser.add_argument("--trajectory-duration", type=float, default=DEFAULT_TRAJECTORY_DURATION,
                        help="inference 轨迹总时长，需与 cascaded_mpc.py 中设置保持一致。")
    parser.add_argument("--results-root", type=Path, default=None,
                        help="可选 inference 结果根目录，默认使用 scripts/inference/logs/mini2。")
    parser.add_argument("--hit-frames", nargs="+", default=list(DEFAULT_HIT_FRAMES),
                        help="需要分别推导碰撞点的 Pinocchio frame 名称。")
    parser.add_argument("--collision-strength", type=float, default=DEFAULT_COLLISION_STRENGTH,
                        help="碰撞强度，范围 [0,1]。1 表示球心穿过 joint，0 表示擦到安全包络边界。")
    parser.add_argument("--safety-margin", type=float, default=DEFAULT_SAFETY_MARGIN,
                        help="用于定义碰撞强度的安全裕度，默认与 cascaded_mpc.py 的 d_safe 一致。")
    parser.add_argument("--quiet", action="store_true",
                        help="只输出每个 joint 对应的动态球初始三维位置。")
    args = parser.parse_args()

    result = compute_dynamic_sphere_position(
        seed=args.seed,
        hit_fraction=args.hit_fraction,
        trajectory_duration=args.trajectory_duration,
        results_root=args.results_root,
        hit_frames=args.hit_frames,
        collision_strength=args.collision_strength,
        safety_margin=args.safety_margin,
    )

    if args.quiet:
        for frame_name, frame_result in result["by_frame"].items():
            print(f"{frame_name}: {_format_array(frame_result['dynamic_sphere_position'])}")
        return

    print(f"seed 编号: {result['seed']}")
    print(f"inference 最优无碰撞轨迹索引: {result['idx_best_traj']}")
    print(f"动态球命中轨迹比例: {result['hit_fraction']:.4f}")
    print(f"动态球命中轨迹步: {result['hit_step']}")
    print(f"动态球命中时间: {result['hit_time']:.6f} s")
    print(f"固定动态球速度: {_format_array(result['dynamic_sphere_velocity'])} m/s")
    print(f"固定动态球半径: {result['dynamic_sphere_radius']:.6f} m")
    print(f"安全裕度: {result['safety_margin']:.6f} m")
    print(f"有效影响半径: {result['effective_radius']:.6f} m")
    print(f"碰撞强度: {result['collision_strength']:.4f}")
    for frame_name, frame_result in result["by_frame"].items():
        position = frame_result["dynamic_sphere_position"]
        print("")
        print(f"[{frame_name}]")
        print(f"joint 目标位置: {_format_array(frame_result['target_position'])}")
        print(f"命中时刻动态球球心位置: {_format_array(frame_result['sphere_position_at_hit'])}")
        print(f"命中时刻球心到 joint 距离: {frame_result['distance_to_joint_at_hit']:.6f} m")
        print("相对有效半径的侵入深度: "
              f"{frame_result['penetration_depth_vs_effective_radius']:.6f} m")
        print(f"动态球初始三维位置: {_format_array(position)}")
        print(f"dynamic_sphere_c0 = np.array({_format_array(position)})")


if __name__ == "__main__":
    main()
