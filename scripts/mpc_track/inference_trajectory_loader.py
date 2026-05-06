# 脚本功能：按 seed 读取 Mini2 inference 结果，并转换为 MPC 可用参考轨迹。
from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin
import torch

try:
    from .asset_paths import mini2_urdf_path, repo_root
except ImportError:
    from asset_paths import mini2_urdf_path, repo_root


@dataclass(frozen=True)
class InferenceTrajectoryReference:
    seed: int
    results_path: Path
    idx_best_traj: Optional[int]
    cost_best_free_traj: Optional[float]
    trajectory_state: np.ndarray
    q_ref: np.ndarray
    v_ref: np.ndarray
    p_ref: np.ndarray
    obstacle_centers: np.ndarray
    obstacle_radii: np.ndarray


MINI2_BASE_OBSTACLE_CENTERS = np.array([
    [-0.25, 0.25, 0.52],
    [-0.30, -0.24, 0.34],
    [-0.34, 0.08, 0.16],
    [0.34, 0.24, 0.16],
    [0.38, 0.18, 0.43],
    [0.42, -0.28, 0.24],
    [0.18, -0.32, 0.43],
    [0.28, 0.05, 0.58],
    [0.05, -0.30, 0.16],
    [0.05, 0.34, 0.30],
], dtype=float)

MINI2_BASE_OBSTACLE_RADII = np.array([
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.09,
    0.09,
    0.09,
    0.09,
], dtype=float)

MINI2_EXTRA_OBSTACLE_CENTERS = np.array([
    [-0.08, 0.14, 0.54],
    [0.26, -0.16, 0.30],
    [-0.22, -0.12, 0.22],
    [0.06, 0.32, 0.60],
], dtype=float)

MINI2_EXTRA_OBSTACLE_RADII = np.array([0.08, 0.08, 0.08, 0.08], dtype=float)


class _TorchCPUUnpickler(pickle.Unpickler):
    """Load torch tensors saved from CUDA checkpoints onto CPU."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(io.BytesIO(data), map_location="cpu")
        return super().find_class(module, name)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _default_results_root() -> Path:
    return repo_root() / "scripts" / "inference" / "logs" / "mini2"


def mini2_inference_obstacles(include_extra_objects: bool = True):
    centers = MINI2_BASE_OBSTACLE_CENTERS
    radii = MINI2_BASE_OBSTACLE_RADII
    if include_extra_objects:
        centers = np.vstack([centers, MINI2_EXTRA_OBSTACLE_CENTERS])
        radii = np.concatenate([radii, MINI2_EXTRA_OBSTACLE_RADII])
    return centers.copy(), radii.copy()


def _load_pickle_cpu(results_path: Path) -> dict:
    with open(results_path, "rb") as handle:
        return _TorchCPUUnpickler(handle).load()


def _select_best_free_trajectory(data: dict):
    best = data.get("traj_final_free_best")
    idx_best = data.get("idx_best_traj")
    if idx_best is not None and isinstance(idx_best, torch.Tensor):
        idx_best = int(idx_best.detach().cpu().item())

    if best is not None:
        return best, idx_best

    free_trajs = data.get("trajs_final_free")
    if free_trajs is None:
        raise ValueError("Inference 结果中没有 traj_final_free_best，也没有 trajs_final_free。")

    n_free = int(free_trajs.shape[0]) if hasattr(free_trajs, "shape") else len(free_trajs)
    if n_free == 0:
        raise ValueError("Inference 结果中没有无碰撞轨迹，无法提供 MPC 跟踪参考。")

    if idx_best is None:
        cost_all = data.get("cost_all_trajs_final_free")
        if cost_all is not None:
            idx_best = int(torch.argmin(torch.as_tensor(_to_numpy(cost_all))).item())
        else:
            idx_best = 0

    return free_trajs[idx_best], int(idx_best)


def _split_state_trajectory(trajectory_state: np.ndarray):
    if trajectory_state.ndim != 2:
        raise ValueError(f"最优轨迹应为二维数组，当前 shape={trajectory_state.shape}")
    if trajectory_state.shape[1] < 6:
        raise ValueError(f"最优轨迹至少需要 6 个关节角维度，当前 shape={trajectory_state.shape}")

    q_ref = trajectory_state[:, :6]
    if trajectory_state.shape[1] >= 12:
        v_ref = trajectory_state[:, 6:12]
    else:
        v_ref = np.zeros_like(q_ref)
    return q_ref.astype(float), v_ref.astype(float)


def _compute_ee_positions(q_ref: np.ndarray, end_frame_name: str):
    model = pin.buildModelFromUrdf(str(mini2_urdf_path()))
    data = model.createData()
    if not model.existFrame(end_frame_name):
        raise RuntimeError(f"找不到末端 frame：{end_frame_name}")

    frame_id = model.getFrameId(end_frame_name)
    positions = []
    for q in q_ref:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        positions.append(np.asarray(data.oMf[frame_id].translation).reshape(3).copy())
    return np.asarray(positions, dtype=float)


def load_inference_reference(
    seed: int,
    results_root: Optional[Path] = None,
    end_frame_name: str = "6_joint",
    include_extra_objects: bool = True,
) -> InferenceTrajectoryReference:
    results_root = Path(results_root).expanduser().resolve() if results_root else _default_results_root()
    results_path = results_root / str(seed) / "results_data_dict.pickle"
    if not results_path.exists():
        raise FileNotFoundError(f"找不到 seed={seed} 的 inference 结果：{results_path}")

    data = _load_pickle_cpu(results_path)
    best_traj, idx_best = _select_best_free_trajectory(data)
    trajectory_state = _to_numpy(best_traj).astype(float)
    q_ref, v_ref = _split_state_trajectory(trajectory_state)
    p_ref = _compute_ee_positions(q_ref, end_frame_name=end_frame_name)
    obstacle_centers, obstacle_radii = mini2_inference_obstacles(
        include_extra_objects=include_extra_objects
    )

    return InferenceTrajectoryReference(
        seed=seed,
        results_path=results_path,
        idx_best_traj=idx_best,
        cost_best_free_traj=_to_optional_float(data.get("cost_best_free_traj")),
        trajectory_state=trajectory_state,
        q_ref=q_ref,
        v_ref=v_ref,
        p_ref=p_ref,
        obstacle_centers=obstacle_centers,
        obstacle_radii=obstacle_radii,
    )
