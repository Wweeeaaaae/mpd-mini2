# 脚本功能：统一管理 mpc_track 中 Mini2 的 URDF、MuJoCo XML 等资源路径。
"""Shared resource path helpers for the MPC tracking scripts."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def mini2_asset_root() -> Path:
    """Return the Mini2 resource directory used by Pinocchio and MuJoCo.

    Set ``MINI2_ASSET_ROOT`` to override the default path. The default points
    to the Mini2 assets already vendored in this repository.
    """

    env_path = os.environ.get("MINI2_ASSET_ROOT")
    if env_path:
        root = Path(env_path).expanduser().resolve()
    else:
        root = repo_root() / "deps" / "torch_robotics" / "torch_robotics" / "data" / "urdf" / "robots" / "Mini2"

    if not root.exists():
        raise FileNotFoundError(
            f"Mini2 resource directory not found: {root}. "
            "Set MINI2_ASSET_ROOT to the directory that contains urdf/, xml/, and meshes/."
        )
    return root


def mini2_urdf_path() -> Path:
    path = mini2_asset_root() / "urdf" / "Mini2.urdf"
    if not path.exists():
        raise FileNotFoundError(f"Mini2 URDF not found: {path}")
    return path


def mini2_scene_path() -> Path:
    path = mini2_asset_root() / "xml" / "scene.xml"
    if not path.exists():
        raise FileNotFoundError(f"Mini2 MuJoCo scene not found: {path}")
    return path
