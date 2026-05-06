# 脚本功能：集中计算 MPC 后验指标并保存跟踪图，避免主 MPC 脚本继续膨胀。
from __future__ import annotations

from pathlib import Path

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


def _summarize_scalar_series(values):
    # 汇总一维标量序列，主要用于 IPOPT 每次 solve 的耗时统计。
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _summarize_error_norms(errors):
    # 输入是按列排列的误差序列，先对每个时间点求范数，再给出常用统计量。
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        return {
            "mean": None,
            "rmse": None,
            "max": None,
            "p95": None,
            "final": None,
        }

    norms = np.linalg.norm(errors, axis=0)
    return {
        "mean": float(np.mean(norms)),
        "rmse": float(np.sqrt(np.mean(norms ** 2))),
        "max": float(np.max(norms)),
        "p95": float(np.percentile(norms, 95)),
        "final": float(norms[-1]),
    }


def _interpolate_reference(reference, time_ref, time_query):
    # 级联 MPC 的实际轨迹更密，这里把 inference 参考插值到同一个时间轴。
    reference = np.asarray(reference, dtype=float)
    time_query = np.clip(np.asarray(time_query, dtype=float), time_ref[0], time_ref[-1])
    return np.vstack([
        np.interp(time_query, time_ref, reference_dim)
        for reference_dim in reference
    ])


def _max_bound_violation(values, lower, upper):
    # 返回所有维度、所有时间点里最大的边界违反量；完全满足约束时为 0。
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    lower = np.asarray(lower, dtype=float).reshape(-1, 1)
    upper = np.asarray(upper, dtype=float).reshape(-1, 1)
    lower_violation = np.maximum(lower - values, 0.0)
    upper_violation = np.maximum(values - upper, 0.0)
    return float(max(np.max(lower_violation), np.max(upper_violation)))


def _compute_static_obstacle_metrics(avoid_points, centers, radii, safety_margin):
    # avoid_points: (n_pts, T, 3)，对所有关键点、所有时刻、所有静态球障碍取最小余量。
    avoid_points = np.asarray(avoid_points, dtype=float)
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    distances = np.linalg.norm(
        avoid_points[:, :, None, :] - centers[None, None, :, :],
        axis=-1,
    )
    surface_clearance = distances - radii[None, None, :]
    low_margin_clearance = distances - (radii + 0.5 * safety_margin)[None, None, :]
    high_margin_clearance = distances - (radii + safety_margin)[None, None, :]
    return {
        "min_surface_clearance_m": float(np.min(surface_clearance)),
        "min_low_margin_clearance_m": float(np.min(low_margin_clearance)),
        "min_high_margin_clearance_m": float(np.min(high_margin_clearance)),
        "low_margin_satisfied": bool(np.min(low_margin_clearance) >= -1e-6),
        "high_margin_satisfied": bool(np.min(high_margin_clearance) >= -1e-6),
    }


def _compute_dynamic_obstacle_metrics(avoid_points, time_actual, center0, velocity, radius, safety_margin):
    # 动态球按 c(t)=c0+v*t 还原每个实际执行时刻的球心，再统计最小余量。
    avoid_points = np.asarray(avoid_points, dtype=float)
    centers = center0.reshape(1, 3) + time_actual.reshape(-1, 1) * velocity.reshape(1, 3)
    distances = np.linalg.norm(avoid_points - centers[None, :, :], axis=-1)
    surface_clearance = distances - radius
    low_margin_clearance = distances - (radius + 0.5 * safety_margin)
    high_margin_clearance = distances - (radius + safety_margin)
    return {
        "min_surface_clearance_m": float(np.min(surface_clearance)),
        "min_low_margin_clearance_m": float(np.min(low_margin_clearance)),
        "min_high_margin_clearance_m": float(np.min(high_margin_clearance)),
        "low_margin_satisfied": bool(np.min(low_margin_clearance) >= -1e-6),
        "high_margin_satisfied": bool(np.min(high_margin_clearance) >= -1e-6),
    }


def _compute_avoid_points_actual(X_total, p_fun_list, n_q):
    # 使用 CasADi FK 函数把实际关节轨迹转换为多个避障关键点轨迹。
    avoid_points_actual_l = []
    for k in range(X_total.shape[1]):
        qk = X_total[0:n_q, k]
        avoid_points_k = []
        for p_fun in p_fun_list:
            pk = np.array(p_fun(ca.DM(qk))).astype(float).flatten()
            avoid_points_k.append(pk)
        avoid_points_actual_l.append(avoid_points_k)
    return np.asarray(avoid_points_actual_l, dtype=float).transpose(1, 0, 2)


def build_mpc_run_metrics(
    *,
    X_total,
    U_total,
    p_fun_list,
    ee_index,
    n_q,
    n_v,
    n_x,
    n_u,
    p_ref,
    x_ref,
    dt,
    dt_low,
    use_dynamic_obstacle,
    use_cascaded_mpc,
    q_min,
    q_max,
    v_min,
    v_max,
    tau_min,
    tau_max,
    acc_min,
    acc_max,
    obstacle_centers,
    obstacle_radii,
    d_safe,
    mpc_success,
    dynamic_sphere_c0=None,
    dynamic_sphere_v=None,
    dynamic_sphere_radius=None,
    single_layer_solve_times=None,
    high_level_solve_times=None,
    low_level_solve_times=None,
):
    # 主入口：根据执行后的 X/U 和参考轨迹，统一生成精简后的关键指标字典。
    X_total = np.asarray(X_total, dtype=float)
    U_total = np.asarray(U_total, dtype=float)
    if U_total.ndim == 1 and U_total.size == 0:
        U_total = np.empty((n_u, 0), dtype=float)

    dt_actual = dt_low if use_dynamic_obstacle and use_cascaded_mpc else dt
    time_actual = np.arange(X_total.shape[1]) * dt_actual
    time_ref = np.arange(np.asarray(p_ref).shape[1]) * dt

    avoid_points_actual = _compute_avoid_points_actual(X_total, p_fun_list, n_q)
    p_traj = avoid_points_actual[ee_index].T

    q_actual = X_total[0:n_q, :]
    v_actual = X_total[n_q:n_x, :]
    acc_actual = (
        np.diff(v_actual, axis=1) / dt_actual
        if v_actual.shape[1] > 1
        else np.empty((n_v, 0), dtype=float)
    )

    q_limit_violation = _max_bound_violation(q_actual, q_min, q_max)
    v_limit_violation = _max_bound_violation(v_actual, v_min, v_max)
    tau_limit_violation = _max_bound_violation(U_total, tau_min, tau_max)
    acc_limit_violation = _max_bound_violation(acc_actual, acc_min, acc_max)
    constraint_tolerance = 1e-6
    constraints_ok = bool(
        max(q_limit_violation, v_limit_violation, tau_limit_violation, acc_limit_violation)
        <= constraint_tolerance
    )

    p_ref_actual = _interpolate_reference(p_ref, time_ref, time_actual)
    x_ref_actual = _interpolate_reference(x_ref, time_ref, time_actual)
    ee_position_error_m = _summarize_error_norms(p_traj - p_ref_actual)
    joint_position_error_rad = _summarize_error_norms(q_actual - x_ref_actual[0:n_q, :])

    seed_obstacle_metrics = _compute_static_obstacle_metrics(
        avoid_points_actual,
        obstacle_centers,
        obstacle_radii,
        d_safe,
    )
    dynamic_obstacle_metrics = None
    if use_dynamic_obstacle:
        dynamic_obstacle_metrics = _compute_dynamic_obstacle_metrics(
            avoid_points_actual,
            time_actual,
            np.asarray(dynamic_sphere_c0, dtype=float),
            np.asarray(dynamic_sphere_v, dtype=float),
            float(dynamic_sphere_radius),
            d_safe,
        )
    obstacle_clearance_ok = bool(
        seed_obstacle_metrics["low_margin_satisfied"]
        and (
            not use_dynamic_obstacle
            or dynamic_obstacle_metrics["low_margin_satisfied"]
        )
    )

    single_layer_solve_times = single_layer_solve_times or []
    high_level_solve_times = high_level_solve_times or []
    low_level_solve_times = low_level_solve_times or []
    all_solver_calls_time_s = _summarize_scalar_series(
        single_layer_solve_times + high_level_solve_times + low_level_solve_times
    )

    verification_passed = bool(
        mpc_success
        and constraints_ok
        and obstacle_clearance_ok
    )

    key_metrics = {
        "ee_position_rmse_m": ee_position_error_m["rmse"],
        "ee_position_max_error_m": ee_position_error_m["max"],
        "joint_position_rmse_rad": joint_position_error_rad["rmse"],
        "seed_obstacle_min_surface_clearance_m": seed_obstacle_metrics["min_surface_clearance_m"],
        "dynamic_obstacle_min_surface_clearance_m": (
            dynamic_obstacle_metrics["min_surface_clearance_m"] if dynamic_obstacle_metrics else None
        ),
        "max_q_limit_violation_rad": q_limit_violation,
        "max_v_limit_violation_rad_s": v_limit_violation,
        "max_tau_limit_violation_nm": tau_limit_violation,
        "solver_mean_time_s": all_solver_calls_time_s["mean"],
        "solver_p95_time_s": all_solver_calls_time_s["p95"],
    }

    return {
        "time_actual": time_actual,
        "time_ref": time_ref,
        "dt_actual": float(dt_actual),
        "avoid_points_actual": avoid_points_actual,
        "p_traj": p_traj,
        "key_metrics": key_metrics,
        "verification_passed": verification_passed,
    }


def build_mpc_run_metrics_from_context(ctx):
    # 从主脚本上下文字典里提取后处理所需字段，避免主脚本展开很长的参数列表。
    return build_mpc_run_metrics(
        X_total=ctx["X_total"],
        U_total=ctx["U_total"],
        p_fun_list=ctx["p_fun_list"],
        ee_index=ctx["ee_index"],
        n_q=ctx["n_q"],
        n_v=ctx["n_v"],
        n_x=ctx["n_x"],
        n_u=ctx["n_u"],
        p_ref=ctx["p_ref"],
        x_ref=ctx["x_ref"],
        dt=ctx["dt"],
        dt_low=ctx["dt_low"],
        use_dynamic_obstacle=ctx["use_dynamic_obstacle"],
        use_cascaded_mpc=ctx["use_cascaded_mpc"],
        q_min=ctx["q_min"],
        q_max=ctx["q_max"],
        v_min=ctx["v_min"],
        v_max=ctx["v_max"],
        tau_min=ctx["tau_min"],
        tau_max=ctx["tau_max"],
        acc_min=ctx["acc_min"],
        acc_max=ctx["acc_max"],
        obstacle_centers=ctx["obstacle_centers"],
        obstacle_radii=ctx["obstacle_radii"],
        d_safe=ctx["d_safe"],
        mpc_success=ctx["mpc_success"],
        dynamic_sphere_c0=ctx["dynamic_sphere_c0"],
        dynamic_sphere_v=ctx["dynamic_sphere_v"],
        dynamic_sphere_radius=ctx["dynamic_sphere_radius"],
        single_layer_solve_times=ctx["single_layer_solve_times"],
        high_level_solve_times=ctx["high_level_solve_times"],
        low_level_solve_times=ctx["low_level_solve_times"],
    )


def plot_mpc_tracking_results(
    *,
    log_dir,
    X_total,
    U_total,
    p_ref,
    q_ref,
    p_traj,
    time_actual,
    time_ref,
    n_q,
    n_u,
    show=False,
):
    # 统一保存末端位置、关节力矩、关节角三类图。
    log_dir = Path(log_dir)
    X_total = np.asarray(X_total, dtype=float)
    U_total = np.asarray(U_total, dtype=float)
    if U_total.ndim == 1 and U_total.size == 0:
        U_total = np.empty((n_u, 0), dtype=float)
    q_ref = np.asarray(q_ref, dtype=float)
    if q_ref.shape[0] != n_q:
        q_ref = q_ref.T

    # 图1：末端位置跟踪。
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    labels = ["X", "Y", "Z"]
    for i in range(3):
        axs1[i].plot(time_actual, p_traj[i, :], label=f"Track Terminal {labels[i]}", color="b")
        axs1[i].plot(time_ref, p_ref[i, :], "--", label=f"Ref Terminal {labels[i]}", color="r")
        axs1[i].set_title(f"Terminal {labels[i]} Position Tracking")
        axs1[i].set_xlabel("Time (s)")
        axs1[i].set_ylabel(f"{labels[i]} Position (m)")
        axs1[i].legend()
        axs1[i].grid(True)
    fig1_path = log_dir / "terminal_position_tracking.png"
    fig1.savefig(fig1_path, dpi=200)

    # 图2：关节力矩。
    fig2, axs2 = plt.subplots(n_u, 1, figsize=(10, 2 * n_u), constrained_layout=True)
    if n_u == 1:
        axs2 = [axs2]
    time_u = time_actual[:U_total.shape[1]]
    for i in range(n_u):
        axs2[i].plot(time_u, U_total[i, :], label=f"Joint {i + 1} Torque", color="b")
        axs2[i].set_title(f"Joint {i + 1} Torque vs Time")
        axs2[i].set_xlabel("Time (s)")
        axs2[i].set_ylabel("Torque (Nm)")
        axs2[i].grid(True)
        axs2[i].legend()
    fig2_path = log_dir / "joint_torque.png"
    fig2.savefig(fig2_path, dpi=200)

    # 图3：关节角。蓝线是 MPC 实际执行角度，红色虚线是 inference 原始角度参考。
    fig3, axs3 = plt.subplots(n_q, 1, figsize=(10, 2 * n_q), constrained_layout=True)
    if n_q == 1:
        axs3 = [axs3]
    for i in range(n_q):
        axs3[i].plot(time_actual, X_total[i, :], label=f"Track Joint {i + 1} Angle", color="b")
        axs3[i].plot(time_ref, q_ref[i, :], "--", label=f"Original Joint {i + 1} Angle", color="r")
        axs3[i].set_title(f"Joint {i + 1} Angle Tracking")
        axs3[i].set_xlabel("Time (s)")
        axs3[i].set_ylabel("Angle (rad)")
        axs3[i].grid(True)
        axs3[i].legend()
    fig3_path = log_dir / "joint_angle_tracking.png"
    fig3.savefig(fig3_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

    return {
        "terminal_position_tracking": str(fig1_path),
        "joint_torque": str(fig2_path),
        "joint_angle_tracking": str(fig3_path),
    }


def plot_mpc_tracking_results_from_context(ctx, metrics_data, show=False):
    # 从上下文直接保存三张图，关节角图会额外叠加 inference 原始关节角参考。
    return plot_mpc_tracking_results(
        log_dir=ctx["log_dir"],
        X_total=ctx["X_total"],
        U_total=ctx["U_total"],
        p_ref=ctx["p_ref"],
        q_ref=ctx["q_ref_np"],
        p_traj=metrics_data["p_traj"],
        time_actual=metrics_data["time_actual"],
        time_ref=metrics_data["time_ref"],
        n_q=ctx["n_q"],
        n_u=ctx["n_u"],
        show=show,
    )


def print_mpc_metrics_summary(metrics_data):
    # 控制台只打印最关键的几项，完整关键指标写入 run_summary.yaml。
    key_metrics = metrics_data["key_metrics"]
    print(
        "[Metrics] "
        f"ee_rmse={key_metrics['ee_position_rmse_m']:.6f} m, "
        f"ee_max={key_metrics['ee_position_max_error_m']:.6f} m, "
        f"seed_min_surface_clearance={key_metrics['seed_obstacle_min_surface_clearance_m']:.6f} m, "
        f"verification_passed={metrics_data['verification_passed']}"
    )


def build_run_summary(ctx, metrics_data):
    # 统一生成 run_summary.yaml 的内容，主脚本只负责写文件。
    use_dynamic_obstacle = ctx["use_dynamic_obstacle"]
    dynamic_obstacle_info = ctx["dynamic_obstacle_info"]
    X_total = ctx["X_total"]
    U_total = ctx["U_total"]

    return {
        "inference_seed": ctx["inference_seed"],
        "mpc_type": ctx["mpc_type"],
        "use_dynamic_obstacle": bool(use_dynamic_obstacle),
        "use_cascaded_mpc": bool(ctx["use_cascaded_mpc"]),
        "dynamic_obstacle_start_position": (
            ctx["dynamic_sphere_c0"].astype(float).tolist() if use_dynamic_obstacle else None
        ),
        "dynamic_obstacle_velocity": (
            ctx["dynamic_sphere_v"].astype(float).tolist() if use_dynamic_obstacle else None
        ),
        "dynamic_obstacle_radius": float(ctx["dynamic_sphere_radius"]) if use_dynamic_obstacle else None,
        "dynamic_obstacle_hit_fraction": (
            float(ctx["dynamic_obstacle_hit_fraction"]) if use_dynamic_obstacle else None
        ),
        "dynamic_obstacle_hit_frame": ctx["dynamic_obstacle_hit_frame"] if use_dynamic_obstacle else None,
        "dynamic_obstacle_collision_strength": (
            float(ctx["dynamic_obstacle_collision_strength"]) if use_dynamic_obstacle else None
        ),
        "dynamic_obstacle_hit_step": int(dynamic_obstacle_info["hit_step"]) if dynamic_obstacle_info else None,
        "dynamic_obstacle_hit_time_s": float(dynamic_obstacle_info["hit_time"]) if dynamic_obstacle_info else None,
        "mpc_total_time_s": float(ctx["mpc_total_time"]),
        "mpc_avg_time_s_per_step": float(ctx["mpc_avg_time"]),
        "steps_expected": int(ctx["N"] - 1),
        "states_recorded": int(X_total.shape[1]),
        "controls_recorded": int(U_total.shape[1]) if U_total.size else 0,
        "success": bool(ctx["mpc_success"]),
        "verification_passed": bool(metrics_data["verification_passed"]),
        "solver_failure_count": int(ctx["solver_failure_count"]),
        "single_layer_failure_count": int(ctx["single_layer_failure_count"]),
        "high_level_failure_count": int(ctx["high_level_failure_count"]),
        "low_level_failure_count": int(ctx["low_level_failure_count"]),
        "key_metrics": metrics_data["key_metrics"],
    }
