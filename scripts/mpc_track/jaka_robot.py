# 脚本功能：封装 Mini2 的 Pinocchio 正逆运动学和动力学计算接口。
"""
特点：
1) 从 URDF 构建模型
2) 末端帧定位（优先 BODY 类型 frame）
3) 正向运动学 FK：q -> 末端 4x4 齐次矩阵
4) 逆向运动学 IK：迭代式 CLIK + 阻尼最小二乘 DLS
5) 支持 6 维任务空间权重 weights（位置3 + 姿态3）
6) 关节限位检查与裁剪
"""

from __future__ import annotations                                                       # 允许在类型注解中引用尚未定义的类型
from dataclasses import dataclass                                                        # 用 dataclass 表达结果更清晰
from typing import Optional                                                              # 类型注解：可选类型与元组
import numpy as np                                                                       # 数值计算库（矩阵/向量/线性代数）
from pathlib import Path
try:
    from .asset_paths import mini2_urdf_path
except ImportError:
    from asset_paths import mini2_urdf_path
try:                                                                                     # 尝试导入 Pinocchio Python 绑定
    import pinocchio as pin
except ImportError as e:
    raise ImportError("无法导入 pinocchio，请先安装 pinocchio 的 Python 绑定") from e

@dataclass(frozen=True)                                                                  # 不可变结果对象，避免被误改
class IKResult:                                                                          # IK 的返回结果结构
    success: bool                                                                        # 是否求解成功
    q: Optional[np.ndarray] = None                                                       # 成功时的关节解（nq,)
    iters: int = 0                                                                       # 实际迭代次数
    final_error_norm: float = float("inf")                                               # 最终 6D 误差范数（加权后）

class jaka:
    def __init__(self, end_frame_name: str = "6_joint"):                                 # 构造函数：加载 URDF + 选择末端 frame
        urdf_path = mini2_urdf_path()
        self.end_frame_name_ = end_frame_name                                            # 末端 frame 名称（由外部传入或使用默认）
        self.model_ = pin.buildModelFromUrdf(str(urdf_path))
        self.data_ = self.model_.createData()                                            # 创建与模型匹配的数据容器
        self.end_frame_id_ = self._find_end_frame_id(self.end_frame_name_)               # 查找末端 frame 的 id（优先 BODY）
        self.joint_id = 6                                                                # 末端执行器为 joint 6

    def _find_end_frame_id(self, frame_name: str) -> int:                                # 根据 frame 名称定位 frame id
        if not self.model_.existFrame(frame_name):
            raise RuntimeError(f"找不到末端 frame：{frame_name}")

        for i, fr in enumerate(self.model_.frames):
            if fr.name == frame_name and fr.type == pin.FrameType.BODY:
                return int(i)

        return int(self.model_.getFrameId(frame_name))                                   # 找不到 BODY 时使用框架提供的兜底查询

    # ============================== 目标姿态矩阵 ============================== #
    def pose_from_pos_quat(self, position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        p = np.asarray(position, dtype=float).reshape(3)
        q = np.asarray(quat_xyzw, dtype=float).reshape(4)

        # 归一化四元数，避免数值漂移导致旋转矩阵畸变
        n = np.linalg.norm(q)
        if n < 1e-12:
            raise ValueError("四元数范数接近 0，无法构造姿态")
        q = q / n

        qx, qy, qz, qw = q                                                               # x,y,z,w
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]
        ], dtype=float)
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    # ============================== 正向运动学（FK） ============================== #
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)                                       # 规范化为 1D float 数组
        if q.size != self.model_.nq:
            raise ValueError(f"q 维度应为 {self.model_.nq}，当前为 {q.size}")

        pin.forwardKinematics(self.model_, self.data_, q)                                # 计算正向运动学
        pin.updateFramePlacements(self.model_, self.data_)                               # 更新各 frame 的位姿

        ee = self.data_.oMf[self.end_frame_id_]                                          # 读取末端 frame 的世界位姿（SE3）
        position = np.asarray(ee.translation).reshape(3)                                 # (x, y, z)
        R = np.asarray(ee.rotation)
        quat = pin.Quaternion(R)                                                         # Pinocchio 四元数对象
        quat_np = np.array([quat.x, quat.y, quat.z, quat.w], dtype=float)         # (x, y, z, w)
        return position, quat_np

    # ============================== 逆向运动学（IK） ============================== #
    def inverse_kinematics(
        self,
        target_position: np.ndarray,                                                     # 末端位置 (3,)
        target_quat_xyzw: np.ndarray,                                                    # 末端四元数 (4,) 顺序 x,y,z,w
        q_init: np.ndarray,                                                              # 初值 q（warm start）
        weights: Optional[np.ndarray] = None,                                            # 6维权重（默认全 1）
        max_iter: int = 1000,                                                            # 最大迭代次数
        eps: float = 1e-4,                                                               # 收敛阈值
        max_retries: int = 5,                                                            # 随机重试次数
        perturb_range: float = 0.5,                                                      # 每个关节的扰动范围 [-p, p]
        rng: Optional[np.random.Generator] = None,                                       # 可注入随机源，便于复现实验
    ) -> IKResult:

        target_pose = self.pose_from_pos_quat(target_position, target_quat_xyzw)

        q_init = np.asarray(q_init, dtype=float).reshape(-1)                             # 规范化初值
        if q_init.size != self.model_.nq:                                                # 初值维度必须匹配
            raise ValueError(f"q_init 维度应为 {self.model_.nq}，当前为 {q_init.size}")

        w = np.ones(6, dtype=float) if weights is None else np.asarray(weights, dtype=float).reshape(-1)  # 权重默认全 1
        if w.size != 6:                                                                  # 权重必须 6 维
            raise ValueError(f"weights 必须是 6 维，当前为 {w.size}")

        # 1) warm start：直接从 q_init 开始迭代
        warm = self._solve_clik(target_pose, q_init, w, max_iter=max_iter, eps=eps)
        if warm.success:
            return warm

        # 2) random retry：随机扰动初值，尝试跳出局部失败区域
        if rng is None:
            rng = np.random.default_rng()

        for _ in range(max_retries):
            q_seed = q_init.copy()                                                       # 从初值复制一个种子
            q_seed += rng.uniform(-perturb_range, perturb_range, size=self.model_.nq)    # 对每个关节施加均匀扰动
            q_seed = self.clamp_to_joint_limits(q_seed)                                  # 将扰动后的种子裁剪到关节范围

            attempt = self._solve_clik(target_pose, q_seed, w, max_iter=max_iter, eps=eps)  # 用新种子求解
            if attempt.success:
                return attempt

        return warm

    # ============================== 关节限位相关 ============================== #
    def check_joint_limits(self, q: np.ndarray, tol: float = 0.01) -> bool:
        # 检查 q 是否在上下限（允许裕量）
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self.model_.nq:
            raise ValueError(f"q 维度应为 {self.model_.nq}，当前为 {q.size}")

        lower = np.asarray(self.model_.lowerPositionLimit).reshape(-1)
        upper = np.asarray(self.model_.upperPositionLimit).reshape(-1)
        return bool(np.all(q >= (lower - tol)) and np.all(q <= (upper + tol)))

    def clamp_to_joint_limits(self, q: np.ndarray) -> np.ndarray:
        # 返回“裁剪后的新数组”（不修改原输入）
        q = np.asarray(q, dtype=float).reshape(-1).copy()
        lower = np.asarray(self.model_.lowerPositionLimit).reshape(-1)
        upper = np.asarray(self.model_.upperPositionLimit).reshape(-1)
        np.clip(q, lower, upper, out=q)
        return q

    def clamp_to_joint_limits_inplace(self, q: np.ndarray) -> None:
        # in-place 版本：直接修改 q
        qv = np.asarray(q, dtype=float).reshape(-1)
        lower = np.asarray(self.model_.lowerPositionLimit).reshape(-1)
        upper = np.asarray(self.model_.upperPositionLimit).reshape(-1)
        np.clip(qv, lower, upper, out=qv)

    def getNumJoints(self) -> int:
        # 返回模型配置维度 nq（常用于检查输入长度）
        return int(self.model_.nq)

    def getEndFrameName(self) -> str:
        # 返回末端 frame 名称
        return self.end_frame_name_

    # ============================== 核心迭代求解器（CLIK + DLS） ============================== #
    def _solve_clik(
        self,
        target_pose: np.ndarray,                                                         # 目标 4x4
        q_guess: np.ndarray,                                                             # 初始猜测
        weights: np.ndarray,                                                             # 6维权重
        max_iter: int,                                                                   # 最大迭代次数
        eps: float,                                                                      # 收敛阈值
        dt: float = 0.2,                                                                 # 更新步长（越大收敛快但可能震荡）
        damping: float = 1e-3,                                                           # DLS 阻尼（越大越稳但越“软”）
    ) -> IKResult:
        q = np.asarray(q_guess, dtype=float).reshape(-1).copy()                          # 当前迭代变量（复制避免污染外部）
        target_se3 = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])                    # 将目标 4x4 转成 SE3

        lower = np.asarray(self.model_.lowerPositionLimit).reshape(-1)
        upper = np.asarray(self.model_.upperPositionLimit).reshape(-1)

        final_err = float("inf")                                                         # 保存最终误差
        for i in range(max_iter):                                                        # 迭代循环
            pin.forwardKinematics(self.model_, self.data_, q)
            pin.updateFramePlacements(self.model_, self.data_)

            current_se3: pin.SE3 = self.data_.oMf[self.end_frame_id_]                    # 当前末端位姿
            err_se3 = current_se3.actInv(target_se3)                                     # 位姿误差：从 current 到 target 的相对变换
            err6 = np.asarray(pin.log6(err_se3).vector).reshape(6)                       # 将 SE(3) 误差映射到 6D 向量
            err6 = err6 * weights                                                        # 对误差维度加权/屏蔽

            final_err = float(np.linalg.norm(err6))                                      # 计算误差范数
            if final_err < eps:                                                          # 收敛判断
                return IKResult(True, q.copy(), i + 1, final_err)

            J = pin.computeFrameJacobian(                                                # 计算末端 frame 的 6×nv 雅可比
                self.model_, self.data_, q, self.end_frame_id_, pin.ReferenceFrame.LOCAL
            )
            J = np.asarray(J, dtype=float)

            for k in range(6):                                                           # 按权重屏蔽 Jacobian 某些行
                if weights[k] < 1e-6:                                                    # 近似为 0 说明“禁用该维度”
                    J[k, :] = 0.0                                                        # 清零该行，避免该维度影响求解

            JJt = J @ J.T                                                                # 构造 6×6 的 JJ^T
            JJt += (damping * damping) * np.eye(6)                                       # 加阻尼让矩阵更稳定可逆

            try:
                v = np.linalg.solve(JJt, err6)                                           # 解线性方程 (JJt) v = err6
            except np.linalg.LinAlgError:                                                # 若矩阵奇异/数值崩溃， 返回失败结果
                return IKResult(False, None, i + 1, final_err)

            dq = J.T @ v                                                                 # 得到关节增量 dq（nv 维）
            q = pin.integrate(self.model_, q, dq * dt)                                   # 在配置空间上积分更新 q

            q = np.minimum(np.maximum(q, lower), upper)                                  # 将 q 快速限制到关节上下限
            q = self.clamp_to_joint_limits(q)                                            # 再做一次严格裁剪（防止数值边界问题）

        return IKResult(False, None, max_iter, final_err)

    # ============================== 动力学求解 ============================== #
    def inverse_dynamics(self, q, v, a):
        # 计算逆动力学（RNEA）
        q = np.asarray(q, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        a = np.asarray(a, dtype=float).reshape(-1)
        tau = pin.rnea(self.model_, self.data_, q, v, a)

        # 打印出每个关节的扭矩（单位：N·m）
        #print("Joint torques: " + str(tau))
        return tau

    def get_coriolis_matrix(self, q, v):
        q = np.asarray(q, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        C = pin.computeCoriolisMatrix(self.model_, self.data_, q, v)
        return C

    def get_mass_matrix(self, q):
        q = np.asarray(q, dtype=float).reshape(-1)
        M = pin.crba(self.model_, self.data_, q)
        return M

    def get_gravity_matrix(self, q):
        q = np.asarray(q, dtype=float).reshape(-1)
        G = pin.computeGeneralizedGravity(self.model_, self.data_, q)
        return G
