# 脚本功能：使用改进 DH 参数对 Mini2/JAKA 机械臂做正运动学对照验证。
from roboticstoolbox import DHRobot, RevoluteMDH
from math import pi
from scipy.spatial.transform import Rotation as R
import numpy as np

# 度转弧度函数
def deg2rad(angle_deg):
    return angle_deg * pi / 180

def mdh_R_p(theta, d, a, alpha):
    """
    根据 MDH 参数，返回单个关节相对于上一关节坐标系的
    旋转矩阵 R (3x3) 和平移向量 p (3,)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    R = np.array([ [ct,      -st * ca,   st * sa],
                   [st,       ct * ca,  -ct * sa],
                   [0.0,      sa,        ca     ]  ])

    p = np.array([  a * ct,
                    a * st,
                    d       ])

    return R, p

# MDH参数表
theta_deg = [0,       -90,      90,       180,        180,    180]              # 关节初始角度 offset（度）
d_vals    = [0.187,   0.0,      0.006,    0.2105,     0.0,    0.1233]           # 沿 z 轴的偏距 d（米）
a_vals    = [0.0,     0.0,      0.21,     0.0,        0.0,    0.0]              # 沿 x 轴的连杆长度 a（米）
alpha_deg = [0,       -90,      180,      90,         90,     90]               # 绕 x 轴的扭转角 alpha（度）

'''
for i in range(6):
    R, p = mdh_R_p(deg2rad(theta_deg[i]), d_vals[i], a_vals[i], deg2rad(alpha_deg[i]))
    print(f"\nJoint {i+1} (相对于 Joint {i}):")
    print("R =\n", R)
    print("p =", p)
'''

# JAKA各关节极限（单位：rad）
qlims = [   [-6.2832,   6.2832],
            [-2.1816,   2.1816],
            [-2.2689,   2.2689],
            [-6.2832,   6.2832],
            [-2.0944,   2.0944],
            [-6.2832,   6.2832],   ]

# 构造 RevoluteMDH 链表
links = []
for th_d, d, a, al_d, qlim in zip(theta_deg, d_vals, a_vals, alpha_deg, qlims):
    links.append(
        RevoluteMDH(
            d=d,
            a=a,
            alpha=deg2rad(al_d),
            offset=deg2rad(th_d),  # theta 作为初始偏置
            qlim=qlim
        )
    )

# 定义机器人
JAKAbot_MDH = DHRobot(links, name="JAKAbot_MDH")

# 计算正向运动学
state0 = [0,0,0,0,0,0]
#state0 = [0.7317047140296069, 1.3336300491938256, -0.03233021228825622, -7.043660504359855e-06, -1.7755941994998583, -0.6983265320327944]

# ==========================================
# 获取每一个关节的位置
# ==========================================

# 计算所有坐标系的正向运动学 (返回一个包含 7 个 SE3 对象的数组：Base + 6个关节)
T_all = JAKAbot_MDH.fkine_all(state0)

print("\n--- 每一个关节的位置 ---")
for i, pose in enumerate(T_all):
    # i=0 代表基座，i=1~6 代表第1到第6个连杆坐标系
    if i == 0:
        name = "基座 (Base)"
    elif i == len(T_all) - 1:
        name = f"末端执行器 (Joint {i})"
    else:
        name = f"关节坐标系 {i}"

    pos = pose.t  # 提取 [x, y, z] 平移向量
    print(f"{name: <15} 位置 (m): x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
