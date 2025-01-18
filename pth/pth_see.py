import torch
import os

# (1) 加载模型
model_path = "/home/yuancz/PycharmProjects/mpd-public/data_trained_models/EnvSpheres3D-RobotPanda/checkpoints/ema_model_current.pth"
model = torch.load(model_path)

# (2) 如果需要推理或验证，通常还会加上 model.eval()
# model.eval()

# (3) 确保输出目录存在
output_dir = "/home/yuancz/PycharmProjects/mpd-public/pth"
os.makedirs(output_dir, exist_ok=True)

# (4) 从 model_path 中获取文件名，例如 "ema_model_current.pth"
filename = os.path.basename(model_path)
# 分离扩展名，例如 filename_without_ext = "ema_model_current"
filename_without_ext, _ = os.path.splitext(filename)

# (5) 拼接 "_decode" 并生成完整路径
# 最终输出文件名形如 "ema_model_current_decode.txt"
output_filename = filename_without_ext + "_decode.txt"
output_file_path = os.path.join(output_dir, output_filename)

# (6) 将模型信息写入文件
with open(output_file_path, "w", encoding="utf-8") as f:
    # a. 打印模型结构
    f.write("----- Model Structure -----\n")
    f.write(str(model))
    f.write("\n\n")

    # b. 查看并打印模型的参数（名称和形状）
    f.write("----- Parameter Name & Shape -----\n")
    for name, param in model.named_parameters():
        f.write(f"Parameter name: {name}\n")
        f.write(f"Shape: {param.shape}\n\n")

    # c. 打印每个参数的数值
    f.write("----- Parameter Values -----\n")
    for name, param in model.named_parameters():
        f.write(f"Parameter name: {name}\n")
        f.write(f"{param}\n\n")

print(f"模型信息已写入: {output_file_path}")
