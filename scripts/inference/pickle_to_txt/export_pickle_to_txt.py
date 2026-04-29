#!/usr/bin/env python3
"""
export_pickle_to_txt_full.py

将一个包含规划结果的 results_data_dict.pickle 文件导出为完整、无省略号的 TXT 文本格式。
"""

import pickle
from pathlib import Path
import sys
import numpy as np
import torch

def tensor_to_string(tensor):
    """Convert a torch Tensor to a full string representation."""
    arr = tensor.cpu().numpy()
    np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=6, suppress=True)
    return np.array2string(arr, max_line_width=np.inf, threshold=np.inf)

def export_pickle_to_txt(pickle_path: Path, output_txt_path: Path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    with open(output_txt_path, 'w', encoding='utf-8') as out:
        for key, value in data.items():
            out.write(f"{key}:\n")
            if isinstance(value, torch.Tensor):
                text = tensor_to_string(value)
            elif isinstance(value, np.ndarray):
                np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=6, suppress=True)
                text = np.array2string(value, max_line_width=np.inf, threshold=np.inf)
            else:
                text = str(value)
            out.write(text + "\n\n")


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # 只需在下面修改为你的 .pickle 文件路径：
    # -----------------------------------------------------------------------------
    pickle_file = Path("/home/yuancz/PycharmProjects/mpd-public/data_trained_models/EnvSpheres3D-RobotPanda/results_inference/other/change_0_results_data_dict.pickle")

    # 指定统一的输出目录（可自动创建）
    output_dir = Path("/home/yuancz/PycharmProjects/mpd-public/scripts/inference/pickle_to_txt")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 在输出目录下，使用与 pickle 同名的 .txt 文件
    output_txt = output_dir / (pickle_file.stem + ".txt")

    # -----------------------------------------------------------------------------
    if not pickle_file.exists():
        print(f"Error: Pickle file not found at {pickle_file}", file=sys.stderr)
        sys.exit(1)

    export_pickle_to_txt(pickle_file, output_txt)
    print(f"Export completed: {output_txt}")
