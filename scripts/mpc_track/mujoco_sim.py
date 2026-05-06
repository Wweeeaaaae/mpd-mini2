# 脚本功能：封装 MuJoCo viewer，用于播放关节轨迹、显示障碍物和录制视频。
import mujoco
import numpy as np
import time
import mujoco.viewer
from xml.etree import ElementTree as ET
import cv2
try:
    from .asset_paths import mini2_scene_path
except ImportError:
    from asset_paths import mini2_scene_path

# ========================= 自定义 Viewer 类 =========================
class CustomViewer:
    def __init__(self, model_path=None):
        # ---------- 启动 MuJoCo 被动 viewer ----------
        self.model_path = str(model_path or mini2_scene_path())
        self.model = mujoco.MjModel.from_xml_path(self.model_path)                                              # 从XML文件加载MuJoCo的JAKA场景模型
        self.data = mujoco.MjData(self.model)                                                                   # 根据 model 创建仿真数据对象
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)                                       # 被动模式：外部更新数据，viewer 负责显示
        self.pos = 0.0001                                                                                       # 预留变量（当前未使用）
        self.handle.cam.distance = 1.8                                                                          # 相机距离场景中心的距离（越大视野越远）
        self.handle.cam.azimuth = 0                                                                             # 相机绕 z 轴的水平旋转角（方位）
        self.handle.cam.elevation = -30                                                                         # 相机仰角：负值表示俯视

        # ---------- 获取 mocap 障碍球 body 与 mocap id ----------
        self.obs_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obs_mocap")
        self.obs_mocap_id = self.model.body_mocapid[self.obs_body_id] if self.obs_body_id != -1 else -1
        self.obs_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obs_vis")

        # ---------- 获取末端执行器 body ID ----------
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, '6_Link')          # 根据刚体名称 'left_finger' 查找 body id
        if self.end_effector_id == -1:
            print("Warning: Could not find the end effector with the given name.")

        # ---------- 记录初始关节角 ----------
        self.initial_q = self.data.qpos[:6].copy()                                                              # 机械臂前 6 个自由度为关节角
        print(f"Initial joint positions: {self.initial_q}")
        self.new_q = self.initial_q                                                                             # 用于存储最新求得的关节角

    def _set_dynamic_sphere_visible(self, visible):
        if self.obs_geom_id == -1:
            return
        if visible:
            self.model.geom_rgba[self.obs_geom_id] = np.array([0.1, 0.35, 1.0, 0.35], dtype=np.float32)
        else:
            self.model.geom_rgba[self.obs_geom_id, 3] = 0.0
        if not visible and self.obs_mocap_id != -1:
            self.data.mocap_pos[self.obs_mocap_id] = np.array([0.0, 0.0, -10.0], dtype=np.float64)

    def _to_points(self, points):
        if points is None:
            return np.empty((0, 3), dtype=np.float64)
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        if points.ndim == 1:
            if points.size != 3:
                raise ValueError("轨迹点必须是 3 维坐标")
            return points.reshape(1, 3)
        if points.ndim == 2 and points.shape[1] == 3:
            return points
        if points.ndim == 2 and points.shape[0] == 3:
            return points.T
        else:
            raise ValueError(f"轨迹点数组必须是 N×3，当前形状为 {points.shape}")

    def _draw_static_markers(self, ref_traj=None, obstacle_centers=None, obstacle_radii=None):
        ref_points = self._to_points(ref_traj)

        # 保持和原始脚本一致：参考轨迹和实际轨迹都直接写入 handle.user_scn。
        with self.handle.lock():
            scn = self.handle.user_scn
            scn.ngeom = 0

            if obstacle_centers is not None and obstacle_radii is not None:
                for center, radius in zip(obstacle_centers, obstacle_radii):
                    if scn.ngeom >= scn.maxgeom:
                        return
                    mujoco.mjv_initGeom(
                        scn.geoms[scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([float(radius), 0.0, 0.0], dtype=np.float64),
                        pos=np.array(center, dtype=np.float64),
                        mat=np.eye(3, dtype=np.float64).flatten(),
                        rgba=np.array([1.0, 0.55, 0.05, 0.35], dtype=np.float32),
                    )
                    scn.ngeom += 1

            for point in ref_points:
                if scn.ngeom >= scn.maxgeom:
                    return
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.005, 0.0, 0.0], dtype=np.float64),
                    pos=np.array(point, dtype=np.float64),
                    mat=np.eye(3, dtype=np.float64).flatten(),
                    rgba=np.array([1.0, 0.0, 0.0, 0.4], dtype=np.float32),
                )
                scn.ngeom += 1

    def _show_pre_motion_markers(self, pause_s=0.8):
        self.sync()
        if pause_s > 0:
            time.sleep(pause_s)

    def _append_actual_marker(self, ee_pos):
        with self.handle.lock():
            scn = self.handle.user_scn
            if scn.ngeom >= scn.maxgeom:
                return
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.005, 0.0, 0.0], dtype=np.float64),
                pos=np.array(ee_pos, dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).flatten(),
                rgba=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            )
            scn.ngeom += 1

    def _write_video_frame(self, renderer, video_writer):
        renderer.update_scene(self.data, camera=self.handle.cam)

        scene = renderer.scene
        user_scene = self._get_scene()
        for i in range(user_scene.ngeom):
            if scene.ngeom < scene.maxgeom:
                src_geom = user_scene.geoms[i]
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    type=src_geom.type,
                    size=np.array(src_geom.size, dtype=np.float64),
                    pos=np.array(src_geom.pos, dtype=np.float64),
                    mat=np.array(src_geom.mat, dtype=np.float64).flatten(),
                    rgba=np.array(src_geom.rgba, dtype=np.float32)
                )
                scene.ngeom += 1

        pixels = renderer.render()
        pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        video_writer.write(pixels_bgr)

    # ========================= viewer API 兼容函数 =========================
    def _get_scene(self):
        if hasattr(self.handle, "user_scn"):
            return self.handle.user_scn
        if hasattr(self.handle, "scn"):
            return self.handle.scn
        if hasattr(self.handle, "_scn"):
            return self.handle._scn
        raise AttributeError(f"Handle has no scene. dir={dir(self.handle)}")

    # ========================= viewer 状态与同步接口 =========================
    def is_running(self):
        return self.handle.is_running()                           # 判断 viewer 窗口是否关闭

    def sync(self):
        self.handle.sync()                                        # 将当前仿真数据同步给 viewer 渲染

    # ========================= 相机 / 视口属性封装 =========================
    @property
    def cam(self):
        return self.handle.cam                                    # 暴露 camera 对象，方便外部调整视角

    @property
    def viewport(self):
        return self.handle.viewport                               # 暴露 viewport 信息

    # ========================= 主仿真循环 =========================
    def run_with_q(self, q_list, dt=0.15, ref_traj=None, obstacle_centers=None, obstacle_radii=None,
                   record_video=False, video_name="mpc_simulation.mp4"):
        mujoco_positions = []
        self._set_dynamic_sphere_visible(False)
        if record_video:
            width, height = 1920, 1080
            self.model.vis.global_.offwidth = width
            self.model.vis.global_.offheight = height
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            fps = int(1.0 / dt) if dt > 0 else 30
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
            print(f"[Video] 开始录制视频，保存至: {video_name}，帧率: {fps} FPS")

        if len(q_list) > 0:
            self.data.qpos[:6] = q_list[0]
            mujoco.mj_forward(self.model, self.data)
        self._draw_static_markers(ref_traj=ref_traj, obstacle_centers=obstacle_centers, obstacle_radii=obstacle_radii)
        self._show_pre_motion_markers()

        for q in q_list:
            if not self.is_running():
                break

            self.data.qpos[:6] = q
            mujoco.mj_forward(self.model, self.data)

            ee_pos = self.data.xpos[self.end_effector_id].copy()
            mujoco_positions.append(ee_pos)
            self._append_actual_marker(ee_pos)

            self.sync()
            time.sleep(dt)

            if record_video:
                self._write_video_frame(renderer, video_writer)

        if record_video:
            video_writer.release()
            print(f"[Video] 视频录制成功，已保存至: {video_name}")

        return np.array(mujoco_positions)

    # ========================= 添加可视化几何体 =========================
    def addVisuGeom(self,
                    geoms_pos:np.ndarray,                               # 几何体的位置 [x, y, z]
                    geoms_type:list,                                    # 几何体的类型 sphere, box 等
                    geoms_size:np.ndarray,                              # 几何体的大小
                    geoms_rgba:np.ndarray):                             # 几何体的颜色

        with self.handle.lock():
            scn = self._get_scene()
            now_user_geom_num = scn.ngeom                              # 获取当前场景中几何体的数量
            total_geoms = min(now_user_geom_num + len(geoms_pos), scn.maxgeom)
            scn.ngeom = total_geoms

            for i in range(total_geoms - now_user_geom_num):
                pos = geoms_pos[i]
                rgba = geoms_rgba[i]
                size = geoms_size[i]
                # 处理尺寸
                if len(size) < 3:
                    if len(size) < 2:
                        size = np.concatenate([size, [0.0, 0.0]])
                    else:
                        size = np.concatenate([size, [0.0]])
                # 确定几何体类型
                ob_type_str = geoms_type[i]
                ob_type = mujoco.mjtGeom.mjGEOM_SPHERE                 # 默认类型为球体
                if ob_type_str == "sphere":
                    ob_type = mujoco.mjtGeom.mjGEOM_SPHERE
                elif ob_type_str == "box":
                    ob_type = mujoco.mjtGeom.mjGEOM_BOX
                elif ob_type_str == "capsule":
                    ob_type = mujoco.mjtGeom.mjGEOM_CAPSULE
                elif ob_type_str == "cylinder":
                    ob_type = mujoco.mjtGeom.mjGEOM_CYLINDER
                elif ob_type_str == "ellipsoid":
                    ob_type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
                elif ob_type_str == "mesh":
                    ob_type = mujoco.mjtGeom.mjGEOM_MESH
                else:
                    raise ValueError(f"Unsupported geom type: {ob_type_str}")
                # 初始化几何体
                mujoco.mjv_initGeom(
                    scn.geoms[i + now_user_geom_num],
                    type=ob_type,
                    size=size,
                    pos=pos,
                    mat=np.eye(3).flatten(),
                    rgba=rgba
                )

    # ========================= 添加障碍物 =========================
    def addObstacles(self,
                     obstacles_pos: np.ndarray,                        # 障碍物的位置 [x, y, z]
                     obstacles_type: list,                             # 障碍物的类型
                     obstacles_size: np.ndarray,                       # 障碍物的大小
                     obstacles_rgba: np.ndarray):                      # 障碍物的颜色

        self.original_model = self.model                               # 保存原始模型
        self.num_obstacles = len(obstacles_pos)                        # 获取障碍物的数量

        # 解析 XML 树（root 是 XML 的根节点）
        tree = ET.parse(self.model_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("原始 XML 中未找到 <worldbody> 标签，请检查 MuJoCo 模型格式")

        for i in range(self.num_obstacles):
            pos = obstacles_pos[i]
            rgba = obstacles_rgba[i]
            size = obstacles_size[i]
            ob_type = obstacles_type[i]

            # 创建 <geom> 节点，添加到 <worldbody> 下
            obstacle_geom = ET.SubElement(worldbody, "geom")
            obstacle_geom.set("name", f"obstacle_{i}")                              # 障碍物名称
            obstacle_geom.set("type", ob_type)                                            # 几何类型
            obstacle_geom.set("size", " ".join(f"{x:.3f}" for x in size))
            obstacle_geom.set("pos", f"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")     # 位置（保留3位小数）
            obstacle_geom.set("contype", "1")                                       # 碰撞类型（必须，与 conaffinity 匹配）
            obstacle_geom.set("conaffinity", "1")                                   # 碰撞亲和性（必须，1=可与同类碰撞）
            # obstacle_geom.set("active", "1")                                                # 启用碰撞/渲染（1=激活）
            obstacle_geom.set("mass", "0.0")                                        # 静态物体（质量=0，不会被推动）
            obstacle_geom.set("rgba", f"{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} {rgba[3]:.3f}")

        # 保存新的xml文件
        new_xml_path = self.model_path.replace(".xml", "_with_obstacles.xml")
        tree.write(new_xml_path, encoding="utf-8", xml_declaration=True)

        # 加载新的模型
        self.model = mujoco.MjModel.from_xml_path(new_xml_path)
        self.data = mujoco.MjData(self.model)

        # 验证是否添加成功
        print(f"原始 Geom 数：{self.original_model.ngeom}")
        print(f"新模型 Geom 数：{self.model.ngeom}")

    # ========================= MPC动态避障仿真 =========================
    def run_with_q_and_moving_sphere(self, q_list, c0, v_obs, dt_vis, dt_sim=0.15, ref_traj=None,
                                     obstacle_centers=None, obstacle_radii=None,
                                     record_video=False, video_name="mpc_simulation.mp4"):
        """
        q_list: (T,6) 关节序列
        c0: (3,) 初始球心
        v_obs: (3,) 球速度
        dt_sim: 仿真步长
        record_video: 是否开启录屏
        video_name: 保存的 MP4 文件名
        """
        if self.obs_mocap_id == -1:
            raise RuntimeError("MuJoCo XML 中未找到 obs_mocap，无法运行动态球仿真")
        self._set_dynamic_sphere_visible(True)

        # ---------- 0. 初始化离屏渲染器和视频写入器 ----------
        if record_video:
            # 设定视频分辨率 (1920x1080)
            width, height = 1920, 1080

            # 强制扩大 MuJoCo 模型的离屏缓冲区大小，防止超过默认上限
            self.model.vis.global_.offwidth = width
            self.model.vis.global_.offheight = height

            # 初始化 MuJoCo 渲染器
            renderer = mujoco.Renderer(self.model, height=height, width=width)

            # 计算帧率：因为 time.sleep(dt_vis)，物理帧率近似为 1/dt_vis
            fps = int(1.0 / dt_vis) if dt_vis > 0 else 30
            # 使用 mp4v 编码
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
            print(f"[Video] 开始录制视频，保存至: {video_name}，帧率: {fps} FPS")

        # ---------- 1. 仿真开始前：一次性画出障碍物和完整红色参考轨迹 ----------
        if len(q_list) > 0:
            self.data.qpos[:6] = q_list[0]
        self.data.mocap_pos[self.obs_mocap_id] = c0
        mujoco.mj_forward(self.model, self.data)
        self._draw_static_markers(ref_traj=ref_traj, obstacle_centers=obstacle_centers, obstacle_radii=obstacle_radii)
        self._show_pre_motion_markers()

        # ---------- 2. 仿真循环：每走一步，追加一个绿色的实际轨迹点 ----------
        mujoco_positions = []
        for k, q in enumerate(q_list):
            if not self.is_running():
                break

            # 更新关节角和障碍球
            self.data.qpos[:6] = q
            c = c0 + v_obs * (k * dt_sim)
            self.data.mocap_pos[self.obs_mocap_id] = c

            # 前向运动学计算
            mujoco.mj_forward(self.model, self.data)

            # 获取真实末端位置
            ee_pos = self.data.xpos[self.end_effector_id].copy()
            mujoco_positions.append(ee_pos)

            # 将当前的实际位置作为绿点追加到场景中
            self._append_actual_marker(ee_pos)

            # 刷新显示并延时
            self.sync()
            time.sleep(dt_vis)

            # ---------- 3. 抓取当前帧并写入视频 ----------
            if record_video:
                self._write_video_frame(renderer, video_writer)

        # ---------- 4. 循环结束后释放视频资源 ----------
        if record_video:
            video_writer.release()
            print(f"[Video] 视频录制成功，已保存至: {video_name}")

        return np.array(mujoco_positions)
