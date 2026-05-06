# 脚本功能：对参考轨迹逐点求 IK，并比较 Pinocchio/MuJoCo 末端位置误差。
import ast
from pathlib import Path
try:
    from .jaka_robot import jaka
    from .mujoco_sim import CustomViewer
except ImportError:
    from jaka_robot import jaka
    from mujoco_sim import CustomViewer
import numpy as np
import matplotlib.pyplot as plt

# ---------------- 绘制图形函数 ----------------
def plot_all(origin, jaka_fk, mujoco_fk):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # -------- 图 1：位置误差 --------
    err_jaka = np.linalg.norm(origin - jaka_fk, axis=1)
    err_mj = np.linalg.norm(origin - mujoco_fk, axis=1)

    plt.figure()
    plt.plot(err_jaka, label='JAKA 计算误差')
    plt.plot(err_mj, label='MuJoCo 仿真误差')
    plt.title('末端位置误差')
    plt.xlabel('时间步')
    plt.ylabel('误差 (m)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # -------- 图 2：XYZ 对比 --------
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    labels = ['X', 'Y', 'Z']
    for i in range(3):
        axs[i].plot(origin[:, i], '--', label='原始位置')
        axs[i].plot(mujoco_fk[:, i], label='MuJoCo位置')
        axs[i].set_ylabel(labels[i] + ' (m)')
        axs[i].legend()
        axs[i].grid()

    axs[-1].set_xlabel('时间步')
    plt.tight_layout()
    plt.show()

# ---------------- 与机器人初始化 ----------------
jaka_robot = jaka()                                                 # 创建 Panda 机器人实例

# ------------- 读取轨迹并做逆运动学 --------------
file_path = Path(__file__).parent / 'trajectory_data' / 'interpolation.txt'
with open(file_path, 'r') as f:                                     # 读取文件所有行
    lines = f.readlines()
first_array = None
for line in lines:                                                  # 找到第一个以 '[' 开头的非空行，并将其解析为 Python 列表
    stripped = line.strip()
    if stripped.startswith('['):
        first_array = ast.literal_eval(stripped)
        break
if first_array is None:
    raise ValueError("文件中未找到任何数组")

angles_buffer = []                                                  # 逆运动学获取各关节角度
origin_buffer = []                                                  # 存储原末端执行器的实际位置
calculat_buffer = []                                                # 存储通过正运动学计算的末端执行器位置

delta_t = 0.15
q_init = [0,0,0,0,0,0]
weight = [1,1,1,0,0,0]
for current_pos in first_array:
    origin_buffer.append(current_pos)
    result = jaka_robot.inverse_kinematics(current_pos, [1,0,0,0], q_init, weight)
    if not result.success or result.q is None:
        raise RuntimeError("IK 失败：某个点求解不到关节角")              # 或者 continue/用上一次解等策略

    # 取出真正的关节角向量
    joint_angle = np.asarray(result.q, dtype=float)
    angles_buffer.append(joint_angle)

    # 计算通过正运动学得到的末端执行器位置
    calculated_position, _ = jaka_robot.forward_kinematics(joint_angle)  # 获取位置部分，忽略四元数
    calculated_copy = calculated_position.copy()
    calculat_buffer.append(calculated_copy)

    q_init = joint_angle                                            # 用上一次解做下一次初值（更稳、更快）

angles_buffer = np.asarray(angles_buffer)
origin_buffer = np.asarray(origin_buffer)
calculat_buffer = np.asarray(calculat_buffer)

# ---------------- mujoco 仿真 ----------------
viewer = CustomViewer()                                           # 创建自定义 viewer
mujoco_fk_pos = viewer.run_with_q(angles_buffer)

# ------------------  画图 ------------------
plot_all(origin_buffer, calculat_buffer, mujoco_fk_pos)
