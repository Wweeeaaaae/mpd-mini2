# 脚本功能：读取 inference 最优轨迹和 seed 环境障碍，并执行 MPC 跟踪与 MuJoCo 可视化。
import time
from pathlib import Path
import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
import yaml
from scipy.interpolate import CubicSpline                                         # 用于轨迹插值
from mujoco_sim import CustomViewer
from asset_paths import mini2_urdf_path
from inference_trajectory_loader import load_inference_reference
from dynamic_obstacle import compute_dynamic_sphere_position
from metrics_and_plots import (
    build_mpc_run_metrics_from_context,
    build_run_summary,
    plot_mpc_tracking_results_from_context,
    print_mpc_metrics_summary,
)


# =============================== 0) 基本参数 =========================================
inference_seed = 5                                                                # 读取 scripts/inference/logs/mini2/<seed>/results_data_dict.pickle
inference_trajectory_duration = 5.0                                               # inference 默认轨迹总时长
use_inference_timing = True                                                       # 用 inference 点数重设高层 dt
urdf_path = mini2_urdf_path()
avoid_frames = ["2_joint", "3_joint", "4_joint", "5_joint", "6_joint"]            # 多关键点避障列表
ee_index = avoid_frames.index("6_joint")                                          # 末端 frame 索引

use_plot = True                                                                   # 是否画图
record_video= True                                                                # 是否记录仿真视频
video_name="mpc_obstacle_avoidance.mp4"                                           # 定义视频名称
use_cascaded_mpc = True                                                           # 动态障碍启用时，是否使用级联 MPC
use_mujoco_viewer = True                                                          # 是否打开 MuJoCo viewer
use_dynamic_obstacle = True                                                       # 是否启用额外动态球实验障碍与高层重规划；seed 环境障碍约束始终启用
use_potential_field = True                                                        # 高层重规划启用时，是否增加 seed 障碍斥力势场代价

mpc_type = "级联MPC" if use_dynamic_obstacle and use_cascaded_mpc else "单层MPC"
mpc_type_slug = "cascaded" if mpc_type == "级联MPC" else "single"
log_dir = Path(__file__).resolve().parent / "logs" / f"seed{inference_seed}_{mpc_type_slug}"
log_dir.mkdir(parents=True, exist_ok=True)
if record_video:
    video_name = str(log_dir / video_name)

dt          = 0.15                                                                # 高层 MPC 控制周期 (s)
Np          = 16                                                                  # 高层预测视野
low_level_ratio = 5                                                               # inference 模式下底层/高层频率倍数
dt_low = 0.015                                                                    # 底层控制周期 (15ms)
Np_low = 10                                                                       # 底层预测视野
ratio = int(dt / dt_low)                                                          # 底层/高层频率倍数

d_safe = 0.04                                                                     # seed 环境障碍安全裕度
dynamic_sphere_v = np.array([0.06, -0.07, 0.03])                                  # 额外动态球匀速速度
dynamic_sphere_radius = 0.06                                                      # 额外动态球半径
dynamic_obstacle_hit_fraction = 0.4                                               # 动态球命中参考轨迹的归一化时刻，0 为起点，1 为终点
dynamic_obstacle_hit_frame = "5_joint"                                            # 用于反推动态球初始位置的碰撞关节
dynamic_obstacle_collision_strength = 0.8                                         # 碰撞强度，1 表示球心穿过 joint，0 表示擦到安全包络边界
dynamic_obstacle_info = None
dynamic_sphere_c0 = np.zeros(3, dtype=float)                                # 额外动态球起始球心，由 dynamic_obstacle.py 计算

if use_dynamic_obstacle:
    dynamic_obstacle_info = compute_dynamic_sphere_position(
        seed=inference_seed,
        hit_fraction=dynamic_obstacle_hit_fraction,
        trajectory_duration=inference_trajectory_duration,
        hit_frames=(dynamic_obstacle_hit_frame,),
        collision_strength=dynamic_obstacle_collision_strength,
        safety_margin=d_safe,
        dynamic_sphere_velocity=dynamic_sphere_v,
        dynamic_sphere_radius=dynamic_sphere_radius,
    )
    dynamic_sphere_c0 = dynamic_obstacle_info["by_frame"][dynamic_obstacle_hit_frame][
        "dynamic_sphere_position"
    ].astype(float)
    print(
        f"[Info] Dynamic obstacle from seed={inference_seed}, "
        f"hit_frame={dynamic_obstacle_hit_frame}, "
        f"hit_fraction={dynamic_obstacle_hit_fraction:.4f}, "
        f"collision_strength={dynamic_obstacle_collision_strength:.4f}, "
        f"c0={dynamic_sphere_c0}"
    )
dynamic_sphere_r_eff_high = dynamic_sphere_radius + d_safe
dynamic_sphere_r_eff_low = dynamic_sphere_radius + 0.5 * d_safe

eta_rep = 5.0                                                                     # 斥力权重，先小后大调
d0_rep = 0.12                                                                     # 势场作用范围
eps_rep = 1e-6                                                                    # 防止除零

# =============================== 1) 建模：Pinocchio 模型 =================================
model = pin.buildModelFromUrdf(str(urdf_path))                                    # 从 URDF 构建刚体模型
data  = model.createData()
avoid_frame_ids = [model.getFrameId(nm) for nm in avoid_frames]                   # 多关键点 joint 的索引
n_pts = len(avoid_frame_ids)
n_q      = model.nq                                                               # 构型维度（一般 6）
n_v      = model.nv                                                               # 速度/力矩维度（一般 6）
n_x      = n_q + n_v                                                              # 状态维度 x=[q; v]
n_u      = n_v                                                                    # 控制输入维度 u=tau

# =============================== 2) 读取 inference 最优无碰撞参考轨迹 ================================
inference_ref = load_inference_reference(
    seed=inference_seed,
    end_frame_name=avoid_frames[ee_index],
)
p_ref_np = inference_ref.p_ref                                                    # (N,3) 末端位置参考
q_ref_np = inference_ref.q_ref                                                    # (N,6) inference 关节参考
v_ref_np = inference_ref.v_ref                                                    # (N,6) inference 速度参考
obstacle_centers = inference_ref.obstacle_centers                                 # (M,3) seed 环境球障碍中心
obstacle_radii = inference_ref.obstacle_radii                                     # (M,) seed 环境球障碍半径
n_obs = obstacle_radii.shape[0]
r_eff_high = obstacle_radii + d_safe
r_eff_low = obstacle_radii + 0.5 * d_safe
if use_inference_timing:
    dt = inference_trajectory_duration / p_ref_np.shape[0]
    ratio = low_level_ratio
    dt_low = dt / ratio
print(
    f"[Info] Loaded inference reference seed={inference_seed}, "
    f"idx_best={inference_ref.idx_best_traj}, "
    f"points={p_ref_np.shape[0]}, obstacles={n_obs}, dt={dt:.6f}, dt_low={dt_low:.6f}"
)

if p_ref_np.ndim != 2 or p_ref_np.shape[1] != 3:
    raise ValueError("参考轨迹必须是 N×3，每个点形如 [x,y,z]")

N = p_ref_np.shape[0]                                                             # 总步数
p_ref = p_ref_np.T                                                                # (3,N) 方便按列取
x_ref_np = np.hstack([q_ref_np, v_ref_np])
x_ref = x_ref_np.T                                                                # (12,N) inference 状态参考


# =============================== 3) CasADi 符号：多关键点 FK（p_i(q)） =========================
# 仍然用 cpin 做 p(q) 的符号表达（用于 MPC 代价），但 M/C/g 改成“冻结参数”
cmodel = cpin.Model(model)
cdata  = cmodel.createData()

q_sym = ca.SX.sym("q", n_q)                                                       # 关节角（符号）
cpin.forwardKinematics(cmodel, cdata, q_sym)                                      # 更新关节位姿（符号）
cpin.updateFramePlacements(cmodel, cdata)                                         # 更新所有 frame 位姿（符号）

p_fun_list = []
for fid in avoid_frame_ids:
    p_i = cdata.oMf[fid].translation                                              # 末端位置 p(q)（世界坐标系，3x1 符号）
    p_fun_list.append(ca.Function(f"p_fun_{fid}", [q_sym], [p_i]))          # p_fun(q)->p，用于代价/绘图


# =============================== 4) 冻结版动力学：把 M/C/g 当参数 =========================
# 4.1 连续动力学：qdot=v；vdot = M^{-1}(tau - C*v - g)
x_sym   = ca.SX.sym("x",   n_x)
tau_sym = ca.SX.sym("tau", n_u)
M_par   = ca.SX.sym("M", n_v, n_v)                                                # ★冻结惯量矩阵参数
C_par   = ca.SX.sym("C", n_v, n_v)                                                # ★冻结科氏/离心矩阵参数
G_par   = ca.SX.sym("g", n_v)                                                     # ★冻结重力项参数
q_x     = x_sym[0:n_q]
v_x     = x_sym[n_q:n_q+n_v]

q_dot = v_x
v_dot = ca.solve(M_par, (tau_sym - C_par @ v_x - G_par))
x_dot = ca.vertcat(q_dot, v_dot)
f = ca.Function("f", [x_sym, tau_sym, M_par, C_par, G_par], [x_dot])        # 连续动力学函数（带冻结参数）

# 4.2 离散积分器 RK4
def rk4_step(fun_f, xk, uk, dt, Mv, Cv, Gv):
    k1 = fun_f(xk,              uk, Mv, Cv, Gv)
    k2 = fun_f(xk + (dt/2)*k1,  uk, Mv, Cv, Gv)
    k3 = fun_f(xk + (dt/2)*k2,  uk, Mv, Cv, Gv)
    k4 = fun_f(xk + dt*k3,      uk, Mv, Cv, Gv)
    return xk + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

Xk = ca.SX.sym("Xk", n_x)
Uk = ca.SX.sym("Uk", n_u)
Mv = ca.SX.sym("Mv", n_v, n_v)
Cv = ca.SX.sym("Cv", n_v, n_v)
Gv = ca.SX.sym("Gv", n_v)

Xk_next = rk4_step(f, Xk, Uk, dt, Mv, Cv, Gv)
F = ca.Function("F", [Xk, Uk, Mv, Cv, Gv], [Xk_next])                       # 离散动力学（带冻结参数），高层积分器

Xk_next_low = rk4_step(f, Xk, Uk, dt_low, Mv, Cv, Gv)
F_low = ca.Function("F_low", [Xk, Uk, Mv, Cv, Gv], [Xk_next_low])           # 底层积分器

# =============================== 5) 约束与权重 ==================================
# 关节角/速度/力矩约束
q_min = np.array([-2*np.pi, -2.0944, -2.2689, -2*np.pi, -2.0944, -2*np.pi])       # 关节角度范围 [±360°,±120°,±125°,±360°,±120°,±360°]
q_max = np.array([2*np.pi, 2.0944, 2.2689, 2*np.pi, 2.0944, 2*np.pi])
v_min   = -np.pi * np.ones(n_v)                                                   # 关节速度限制 180°/s（3.1416 rad/s）
v_max   =  np.pi * np.ones(n_v)
acc_limit = 400 * np.pi / 180                                                     # 关节加速度限制 400°/s²
acc_min = -acc_limit * np.ones(n_q)
acc_max =  acc_limit * np.ones(n_q)
tau_min = -10 * np.ones(n_u)                                                      # 力矩范围无要求 [-10, 10] Nm
tau_max =  10 * np.ones(n_u)

rho_obs = 1e6                                                                     # 软约束罚权（暂设，越大约束越硬）
eps_norm = 1e-8                                                                   # 平滑norm用

# 高层重规划 MPC 代价权重：末端位置跟踪 + inference 状态跟踪 + 力矩正则
Qp = np.diag([20.0, 20.0, 20.0])                                                  # 末端位置误差权重
R  = np.diag([0.2] * n_u)                                                         # 力矩正则权重
Qq_reg = np.diag([1e-2] * n_q)                                                    # ★关节正则：缓解“6 轴跟 3 维位置”的冗余多解
use_joint_regularization = True
use_reference_state_tracking = True                                               # 跟踪 inference 完整关节状态
Qx_ref_high = np.diag([5.0] * n_q + [0.2] * n_v)

# 单层/底层 MPC 代价权重：状态跟踪 + 力矩正则
Qx_low = np.diag([100.0] * n_q + [5.0] * n_v)
R_low = np.diag([0.1] * n_u)

# =============================== 6) 初始状态 ==============================
q0 = q_ref_np[0].copy()
v0 = v_ref_np[0].copy()
x_current = np.concatenate([q0, v0])                                              # 当前状态 x=[q;v]
X_total   = [x_current.copy()]                                                    # 记录状态轨迹
U_total   = []                                                                    # 记录力矩轨迹

# =============================== 7.A) 提前构建高层重规划 MPC Opti 符号结构  ===============================
if use_dynamic_obstacle:
    print("[Info] Building CasADi Opti structure for High-Level Replanner...")
else:
    print("[Info] Dynamic obstacle disabled: high-level replanning MPC will not be used.")
opti = ca.Opti()

# 7.1 声明决策变量 (Variables)
X = opti.variable(n_x, Np + 1)                                              # 状态轨迹
U = opti.variable(n_u, Np)                                                  # 力矩轨迹
S_obs = opti.variable(n_pts * (n_obs + 1), Np) if use_dynamic_obstacle else None  # seed 环境障碍 + 动态球软约束松弛变量

# 7.2 声明随时间变化的参数 (Parameters)
# 只要是每次 MPC 循环里不一样的值，都必须是 parameter
x0_param = opti.parameter(n_x)                                                    # 当前时刻的初始状态 x_current
P_ref_param = opti.parameter(3, Np)                                         # 预测域内的末端参考轨迹
M_op = opti.parameter(n_v, n_v)                                             # 当前步算出的冻结 M 矩阵
C_op = opti.parameter(n_v, n_v)                                             # 当前步算出的冻结 C 矩阵
G_op = opti.parameter(n_v)                                                        # 当前步算出的冻结 G 向量
C_dyn_param = opti.parameter(3, Np) if use_dynamic_obstacle else None       # 额外动态球在高层预测域内的球心

if use_joint_regularization:
    q_nom_param = opti.parameter(n_q)                                             # 关节正则化的目标位姿
if use_reference_state_tracking:
    X_ref_high_param = opti.parameter(n_x, Np)                              # inference 完整状态参考

# 7.3 构建约束与代价 (符号图构建)
opti.subject_to(X[:, 0] == x0_param)                                              # 将起点绑定为参数
if use_dynamic_obstacle:
    opti.subject_to(ca.vec(S_obs) >= 0)                                           # 松弛变量非负

cost = 0
for k in range(Np):
    xk = X[:, k]
    xnext = X[:, k + 1]
    uk = U[:, k]

    qk = xk[0:n_q]
    vk = xk[n_q:n_q + n_v]
    vnext = xnext[n_q:n_q + n_v]

    # 状态 & 控制约束 (CasADi 支持 opti.bounded 简写)
    opti.subject_to(opti.bounded(q_min, qk, q_max))
    opti.subject_to(opti.bounded(v_min, vk, v_max))
    opti.subject_to(opti.bounded(tau_min, uk, tau_max))

    acc_approx = (vnext - vk) / dt
    opti.subject_to(opti.bounded(acc_min, acc_approx, acc_max))

    # 动力学约束 (使用参数化的 M_op, C_op, G_op)
    x_next_pred = F(xk, uk, M_op, C_op, G_op)
    opti.subject_to(xnext == x_next_pred)

    # 代价函数：末端误差 + inference 状态误差 + 控制力矩
    pk = p_fun_list[ee_index](qk)
    ep = pk - P_ref_param[:, k]                                                   # ★ 使用参数化的 P_ref
    cost += ca.mtimes([ep.T, Qp, ep])
    cost += ca.mtimes([uk.T, R, uk])

    if use_reference_state_tracking:
        ex_high = xk - X_ref_high_param[:, k]
        cost += ca.mtimes([ex_high.T, Qx_ref_high, ex_high])

    if use_dynamic_obstacle:
        # 多关键点 seed 环境障碍 + 额外动态球软约束、可选势场代价（斥力）
        for i in range(n_pts):
            p_i = p_fun_list[i](qk)
            for j in range(n_obs):
                ck = ca.DM(obstacle_centers[j])
                diff = p_i - ck
                dist = ca.sqrt(ca.dot(diff, diff) + eps_norm)
                slack_idx = i * (n_obs + 1) + j
                opti.subject_to(dist + S_obs[slack_idx, k] >= r_eff_high[j])
                cost += rho_obs * (S_obs[slack_idx, k] ** 2)

                if use_potential_field:
                    d_eff = ca.fmax(dist, eps_rep)                                # 防止除零
                    rep_term = 0.5 * eta_rep * ca.fmax(0, 1.0 / d_eff - 1.0 / d0_rep) ** 2
                    cost += rep_term

            ck_dyn = C_dyn_param[:, k]
            diff_dyn = p_i - ck_dyn
            dist_dyn = ca.sqrt(ca.dot(diff_dyn, diff_dyn) + eps_norm)
            slack_idx_dyn = i * (n_obs + 1) + n_obs
            opti.subject_to(dist_dyn + S_obs[slack_idx_dyn, k] >= dynamic_sphere_r_eff_high)
            cost += rho_obs * (S_obs[slack_idx_dyn, k] ** 2)

            if use_potential_field:
                d_eff_dyn = ca.fmax(dist_dyn, eps_rep)
                rep_dyn = 0.5 * eta_rep * ca.fmax(0, 1.0 / d_eff_dyn - 1.0 / d0_rep) ** 2
                cost += rep_dyn

    # 关节正则
    if use_joint_regularization:
        eqr = qk - q_nom_param                                                    # ★ 使用参数化的 q_nom
        cost += ca.mtimes([eqr.T, Qq_reg, eqr])

opti.minimize(cost)

# 7.4 配置求解器
opts = {"ipopt.print_level": 0, "print_time": 0}
opti.solver("ipopt", opts)

# 7.5 缓存上一周期的解，用于暖启动
X_guess_prev = None
U_guess_prev = None
S_guess_prev = None

# =============================== 7.B) 构建底层跟踪 MPC Opti ===============================
if use_dynamic_obstacle and use_cascaded_mpc:
    print("[Info] Building CasADi Opti structure for Low-Level Tracker...")
    opti_low = ca.Opti()

    X_low = opti_low.variable(n_x, Np_low + 1)
    U_low = opti_low.variable(n_u, Np_low)

    x0_low_param = opti_low.parameter(n_x)
    X_ref_param = opti_low.parameter(n_x, Np_low)
    M_low_op = opti_low.parameter(n_v, n_v)
    C_low_op = opti_low.parameter(n_v, n_v)
    G_low_op = opti_low.parameter(n_v)
    C_dyn_low_param = opti_low.parameter(3, Np_low)                               # 额外动态球在底层预测域内的球心

    opti_low.subject_to(X_low[:, 0] == x0_low_param)

    cost_low = 0
    for k in range(Np_low):
        xk, xnext, uk = X_low[:, k], X_low[:, k + 1], U_low[:, k]
        qk, vk, vnext = xk[0:n_q], xk[n_q:n_x], xnext[n_q:n_x]

        # 状态与力矩极限
        opti_low.subject_to(opti_low.bounded(q_min, qk, q_max))
        opti_low.subject_to(opti_low.bounded(v_min, vk, v_max))
        opti_low.subject_to(opti_low.bounded(tau_min, uk, tau_max))
        opti_low.subject_to(opti_low.bounded(acc_min, (vnext - vk) / dt_low, acc_max))

        # 动力学硬约束 (使用底层小步长 F_low)
        opti_low.subject_to(xnext == F_low(xk, uk, M_low_op, C_low_op, G_low_op))

        # seed 环境障碍硬约束
        for i in range(n_pts):
            p_i = p_fun_list[i](qk)
            for j in range(n_obs):
                ck = ca.DM(obstacle_centers[j])
                diff = p_i - ck
                dist = ca.sqrt(ca.sumsqr(diff) + eps_norm)
                opti_low.subject_to(dist >= r_eff_low[j])

            ck_dyn = C_dyn_low_param[:, k]
            diff_dyn = p_i - ck_dyn
            dist_dyn = ca.sqrt(ca.sumsqr(diff_dyn) + eps_norm)
            opti_low.subject_to(dist_dyn >= dynamic_sphere_r_eff_low)

        # 跟踪代价
        ex = xk - X_ref_param[:, k]
        cost_low += ca.mtimes([ex.T, Qx_low, ex]) + ca.mtimes([uk.T, R_low, uk])

    opti_low.minimize(cost_low)
    opti_low.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0,"ipopt.max_iter": 30, "ipopt.tol": 1e-3})

    # 底层暖启动缓存
    X_low_guess, U_low_guess = None, None

# =============================== 7.C) 构建无动态障碍时的单层跟踪 MPC Opti ===============================
if not use_dynamic_obstacle:
    print("[Info] Building CasADi Opti structure for Single-Layer Inference Tracker...")
    opti_track = ca.Opti()

    X_track = opti_track.variable(n_x, Np + 1)
    U_track = opti_track.variable(n_u, Np)
    S_track = opti_track.variable(n_pts * n_obs, Np)

    x0_track_param = opti_track.parameter(n_x)
    X_ref_track_param = opti_track.parameter(n_x, Np)
    M_track_op = opti_track.parameter(n_v, n_v)
    C_track_op = opti_track.parameter(n_v, n_v)
    G_track_op = opti_track.parameter(n_v)

    opti_track.subject_to(X_track[:, 0] == x0_track_param)
    opti_track.subject_to(ca.vec(S_track) >= 0)

    cost_track = 0
    for k in range(Np):
        xk, xnext, uk = X_track[:, k], X_track[:, k + 1], U_track[:, k]
        qk, vk, vnext = xk[0:n_q], xk[n_q:n_x], xnext[n_q:n_x]

        opti_track.subject_to(opti_track.bounded(q_min, qk, q_max))
        opti_track.subject_to(opti_track.bounded(v_min, vk, v_max))
        opti_track.subject_to(opti_track.bounded(tau_min, uk, tau_max))
        opti_track.subject_to(opti_track.bounded(acc_min, (vnext - vk) / dt, acc_max))
        opti_track.subject_to(xnext == F(xk, uk, M_track_op, C_track_op, G_track_op))

        ex = xk - X_ref_track_param[:, k]
        cost_track += ca.mtimes([ex.T, Qx_low, ex]) + ca.mtimes([uk.T, R_low, uk])

        # 单层跟踪仍约束 seed 环境障碍，防止跟踪误差导致碰撞
        for i in range(n_pts):
            p_i = p_fun_list[i](qk)
            for j in range(n_obs):
                ck = ca.DM(obstacle_centers[j])
                diff = p_i - ck
                dist = ca.sqrt(ca.sumsqr(diff) + eps_norm)
                slack_idx = i * n_obs + j
                opti_track.subject_to(dist + S_track[slack_idx, k] >= r_eff_low[j])
                cost_track += rho_obs * (S_track[slack_idx, k] ** 2)

    opti_track.minimize(cost_track)
    opti_track.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 50, "ipopt.tol": 1e-3})

    X_track_guess, U_track_guess = None, None

# =============================== 8) MPC 主循环（冻结 M/C/G） =============================
solver_failure_count = 0
single_layer_failure_count = 0
high_level_failure_count = 0
low_level_failure_count = 0
single_layer_solve_times = []
high_level_solve_times = []
low_level_solve_times = []
t0_mpc = time.perf_counter()

for t in range(N - 1):
    # 8.1 准备本步数据（参考轨迹、可选障碍物预测、动力学参数）
    if t + Np <= N:
        pref_h = p_ref[:, t:t+Np]                                                 # (3,Np)
    else:
        need   = (t + Np) - N
        tail   = p_ref[:, N-1:N]                                                  # (3,1)
        pref_h = np.hstack([p_ref[:, t:N], np.repeat(tail, need, axis=1)])
    if use_reference_state_tracking:
        if t + Np <= N:
            xref_h = x_ref[:, t:t+Np]
        else:
            need = (t + Np) - N
            tail = x_ref[:, N-1:N]
            xref_h = np.hstack([x_ref[:, t:N], np.repeat(tail, need, axis=1)])

    q_t = x_current[0:n_q]
    v_t = x_current[n_q:n_q+n_v]
    M_val = pin.crba(model, data, q_t)                                            # (n_v,n_v) 惯量矩阵
    M_val = (M_val + M_val.T) / 2                                                 # ★数值稳健：强制对称
    C_val = pin.computeCoriolisMatrix(model, data, q_t, v_t)                      # (n_v,n_v) 科氏/离心矩阵
    G_val = pin.computeGeneralizedGravity(model, data, q_t)                       # (n_v,)   重力项

    if not use_dynamic_obstacle:
        # 无动态障碍时不做高层重规划，直接用单层 MPC 跟踪 inference 状态轨迹，并约束 seed 环境障碍
        opti_track.set_value(x0_track_param, x_current)
        opti_track.set_value(X_ref_track_param, xref_h)
        opti_track.set_value(M_track_op, M_val)
        opti_track.set_value(C_track_op, C_val)
        opti_track.set_value(G_track_op, G_val)

        if X_track_guess is None:
            opti_track.set_initial(X_track, np.tile(x_current.reshape(-1, 1), (1, Np + 1)))
            opti_track.set_initial(U_track, np.zeros((n_u, Np)))
            opti_track.set_initial(S_track, np.zeros((n_pts * n_obs, Np)))
        else:
            X_init_track = np.hstack([X_track_guess[:, 1:], X_track_guess[:, -1:]])
            X_init_track[:, 0] = x_current
            U_init_track = np.hstack([U_track_guess[:, 1:], U_track_guess[:, -1:]])
            opti_track.set_initial(X_track, X_init_track)
            opti_track.set_initial(U_track, U_init_track)
            opti_track.set_initial(S_track, np.zeros((n_pts * n_obs, Np)))

        solve_t0 = time.perf_counter()
        try:
            sol_track = opti_track.solve()
            u0_track = sol_track.value(U_track[:, 0])
            X_track_guess = sol_track.value(X_track)
            U_track_guess = sol_track.value(U_track)
        except RuntimeError as e:
            solver_failure_count += 1
            single_layer_failure_count += 1
            print(f"[Warn] single-layer tracker failed at t={t}: {e}")
            if U_track_guess is not None:
                u0_track = U_track_guess[:, 1] if U_track_guess.shape[1] > 1 else U_track_guess[:, 0]
            else:
                u0_track = np.zeros(n_u)
        finally:
            single_layer_solve_times.append(time.perf_counter() - solve_t0)

        x_next = np.array(F(ca.DM(x_current), ca.DM(u0_track), ca.DM(M_val), ca.DM(C_val), ca.DM(G_val))).astype(float).flatten()
        X_total.append(x_next.copy())
        U_total.append(u0_track.copy())
        x_current = x_next
        continue

    # 8.2 向 Opti 结构注入当前时刻的具体数值 (set_value)
    opti.set_value(x0_param, x_current)                                     # 注入当前状态
    opti.set_value(P_ref_param, pref_h)                                     # 注入参考轨迹
    opti.set_value(M_op, M_val)                                             # 注入动力学 M
    opti.set_value(C_op, C_val)                                             # 注入动力学 C
    opti.set_value(G_op, G_val)                                             # 注入动力学 G
    C_dyn_h = np.zeros((3, Np))
    t_now = t * dt
    for k in range(Np):
        C_dyn_h[:, k] = dynamic_sphere_c0 + dynamic_sphere_v * (t_now + k * dt)
    opti.set_value(C_dyn_param, C_dyn_h)                                    # 注入额外动态球全局时间球心
    if use_joint_regularization:
        opti.set_value(q_nom_param, q_t)                                    # 注入正则目标
    if use_reference_state_tracking:
        opti.set_value(X_ref_high_param, xref_h)                            # 注入 inference 状态参考

    # 8.3 暖启动 (set_initial)
    if X_guess_prev is None:
        opti.set_initial(X, np.tile(x_current.reshape(-1, 1), (1, Np + 1)))
        opti.set_initial(U, np.zeros((n_u, Np)))
        opti.set_initial(S_obs, np.zeros((n_pts * (n_obs + 1), Np)))
    else:
        X_init = np.hstack([X_guess_prev[:, 1:], X_guess_prev[:, -1:]])           # 轨迹前移一格
        U_init = np.hstack([U_guess_prev[:, 1:], U_guess_prev[:, -1:]])

        X_init[:, 0] = x_current
        opti.set_initial(X, X_init)
        opti.set_initial(U, U_init)
        if use_dynamic_obstacle:
            S_init = np.hstack([S_guess_prev[:, 1:], S_guess_prev[:, -1:]])
            opti.set_initial(S_obs, S_init)

    # 8.4 高层 MPC 求解与执行
    solve_t0 = time.perf_counter()
    try:
        sol = opti.solve()
        u0  = sol.value(U[:, 0])
        X_guess_prev = sol.value(X)                                               # 保存整段解用于下一次 warm-start
        U_guess_prev = sol.value(U)
        if use_dynamic_obstacle:
            S_guess_prev = sol.value(S_obs)
    except RuntimeError as e:
        solver_failure_count += 1
        high_level_failure_count += 1
        print(f"[Warn] solver failed at t={t}: {e}")
        if U_guess_prev is not None:                                              # 用上一轮的下一步控制作为 fallback（更接近可行轨迹）
            u0 = U_guess_prev[:, 1] if U_guess_prev.shape[1] > 1 else U_guess_prev[:, 0]
        else:
            u0 = np.zeros(n_u)
    finally:
        high_level_solve_times.append(time.perf_counter() - solve_t0)

    # 8.5 环境推演或底层跟踪
    if not use_cascaded_mpc:
        # 【动态障碍单层重规划模式】：直接用重规划 MPC 的第一个力矩推演 dt 秒
        x_next = np.array(F(ca.DM(x_current), ca.DM(u0), ca.DM(M_val), ca.DM(C_val), ca.DM(G_val))).astype(float).flatten()
        X_total.append(x_next.copy())
        U_total.append(u0.copy())
        x_current = x_next
    else:
        # 【级联模式】：插值高层轨迹，运行 ratio 次底层控制
        # 1. 对高层预测出的轨迹 X_guess_prev (n_x, Np+1) 进行样条插值
        t_high_nodes = np.arange(Np + 1) * dt
        spline_X = CubicSpline(t_high_nodes, X_guess_prev, axis=1)

        # 2. 底层运行 ratio 次
        for i_low in range(ratio):
            # 重算当前极短时刻的动力学矩阵
            q_low = x_current[0:n_q]
            v_low = x_current[n_q:n_x]
            M_val_low = (pin.crba(model, data, q_low) + pin.crba(model, data, q_low).T) / 2
            C_val_low = pin.computeCoriolisMatrix(model, data, q_low, v_low)
            G_val_low = pin.computeGeneralizedGravity(model, data, q_low)

            # 准备底层的参考轨迹
            t_eval = i_low * dt_low + np.arange(Np_low) * dt_low
            t_eval = np.clip(t_eval, 0, t_high_nodes[-1])  # 防止样条外推
            X_ref_val = spline_X(t_eval)

            # 注入参数
            opti_low.set_value(x0_low_param, x_current)
            opti_low.set_value(X_ref_param, X_ref_val)
            opti_low.set_value(M_low_op, M_val_low)
            opti_low.set_value(C_low_op, C_val_low)
            opti_low.set_value(G_low_op, G_val_low)
            C_dyn_low_val = np.zeros((3, Np_low))
            t_now_low = t * dt + i_low * dt_low
            for k in range(Np_low):
                C_dyn_low_val[:, k] = dynamic_sphere_c0 + dynamic_sphere_v * (t_now_low + k * dt_low)
            opti_low.set_value(C_dyn_low_param, C_dyn_low_val)

            # 暖启动
            if X_low_guess is None:
                opti_low.set_initial(X_low, np.tile(x_current.reshape(-1, 1), (1, Np_low + 1)))
                opti_low.set_initial(U_low, np.zeros((n_u, Np_low)))
            else:
                X_init_low = np.hstack([X_low_guess[:, 1:], X_low_guess[:, -1:]])
                X_init_low[:, 0] = x_current
                U_init_low = np.hstack([U_low_guess[:, 1:], U_low_guess[:, -1:]])
                opti_low.set_initial(X_low, X_init_low)
                opti_low.set_initial(U_low, U_init_low)

            # 求解底层 MPC
            solve_t0 = time.perf_counter()
            try:
                sol_low = opti_low.solve()
                u0_low = sol_low.value(U_low[:, 0])
                X_low_guess = sol_low.value(X_low)
                U_low_guess = sol_low.value(U_low)
            except RuntimeError:
                solver_failure_count += 1
                low_level_failure_count += 1
                print(f"[Warn] Low-level solver failed at t={t}, i_low={i_low}")
                u0_low = U_low_guess[:, 1] if U_low_guess is not None and U_low_guess.shape[1] > 1 else np.zeros(n_u)
            finally:
                low_level_solve_times.append(time.perf_counter() - solve_t0)

            # 使用底层 u0_low 与 F_low 推演 dt_low 秒
            x_next = np.array(F_low(ca.DM(x_current), ca.DM(u0_low), ca.DM(M_val_low), ca.DM(C_val_low), ca.DM(G_val_low))).astype(float).flatten()
            X_total.append(x_next.copy())
            U_total.append(u0_low.copy())
            x_current = x_next


t1_mpc = time.perf_counter()
mpc_total_time = t1_mpc - t0_mpc
mpc_avg_time = mpc_total_time / (N - 1)
mpc_success = solver_failure_count == 0 and len(X_total) >= N
print(f"[Timing] MPC total: {mpc_total_time:.6f} s, steps={N-1}, avg={mpc_avg_time:.6f} s/step")


# =============================== 9) 后处理与可视化 ========================================
X_total = np.array(X_total).T                                                     # (n_x, N)
U_total = np.array(U_total).T                                                     # (n_u, N-1)

run_context = dict(locals())                                                     # 固化当前运行上下文，后处理模块从这里取指标和绘图所需字段
metrics_data = build_mpc_run_metrics_from_context(run_context)
print_mpc_metrics_summary(metrics_data)

# --- 可选 MuJoCo 仿真环节 ---
if use_dynamic_obstacle and use_cascaded_mpc:
    dt_sim = dt_low                                                              # 更新仿真器的时间步长为底层高频步长
    dt_vis = 0.1
else:
    dt_sim = dt                                                                 # 更新仿真器的时间步长为高层低频步长
    dt_vis = 0.5

q_list = X_total[0:6, :].T                                                      # 从 MPC 拿到关节序列
ref_traj_list = p_ref.T.tolist()                                                # 将参考轨迹 (N*3) 转换为 Python list 传进去

ee_traj = None
if use_mujoco_viewer:
    viewer = CustomViewer()
    if use_dynamic_obstacle:
        ee_traj = viewer.run_with_q_and_moving_sphere(
            q_list,
            dynamic_sphere_c0,
            dynamic_sphere_v,
            dt_vis=dt_vis,
            dt_sim=dt_sim,
            ref_traj=ref_traj_list,
            obstacle_centers=obstacle_centers,
            obstacle_radii=obstacle_radii,
            record_video=record_video,
            video_name=video_name,
        )
    else:
        ee_traj = viewer.run_with_q(
            q_list,
            dt=dt_vis,
            ref_traj=ref_traj_list,
            obstacle_centers=obstacle_centers,
            obstacle_radii=obstacle_radii,
            record_video=record_video,
            video_name=video_name,
        )

if use_plot:
    plot_mpc_tracking_results_from_context(run_context, metrics_data, show=True)

run_summary = build_run_summary(run_context, metrics_data)
summary_yaml_path = log_dir / "run_summary.yaml"
with open(summary_yaml_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(run_summary, f, allow_unicode=True, sort_keys=False)
print(f"[Log] Run summary saved to: {summary_yaml_path}")
