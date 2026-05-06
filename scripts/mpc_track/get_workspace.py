# 脚本功能：随机采样关节空间并绘制 Mini2 末端可达工作空间。
try:
    from .jaka_robot import jaka
    from . import mujoco_sim
except ImportError:
    from jaka_robot import jaka
    import mujoco_sim
import numpy as np                                                 # 导入NumPy（数值计算）
from pathlib import Path
import matplotlib.pyplot as plt                                    # 导入matplotlib绘图
from mpl_toolkits.mplot3d import Axes3D                            # 导入3D坐标轴工具（触发3D支持）
import mujoco                                                      # 导入MuJoCo引擎
try:
    from .asset_paths import mini2_scene_path, mini2_urdf_path
except ImportError:
    from asset_paths import mini2_scene_path, mini2_urdf_path

class GetWorkspace(mujoco_sim.CustomViewer):
    def __init__(self, sample_num=int(10e4)):
        urdf_path = mini2_urdf_path()
        scene_path = mini2_scene_path()
        super().__init__(scene_path)                                 # 初始化父类并设置相机视角
        self.path = scene_path                                      # 保存场景路径
        self.urdf_path = urdf_path                                  # 保存URDF路径
        self.sample_num = sample_num                                # 保存采样数量

    def runBefore(self):
        self.jaka_robot = jaka()
        workspace = []                                              # 存放采样到的末端位置点集

        for _ in range(self.sample_num):                            # 循环进行随机采样
            low = self.jaka_robot.model_.lowerPositionLimit         # 读取该关节下限
            high = self.jaka_robot.model_.upperPositionLimit        # 读取该关节上限
            q = np.random.uniform(low, high)

            position, quat = self.jaka_robot.forward_kinematics(q)  # 末端执行器的位姿（位置+四元数）
            x, y, z = position                                      # 提取末端执行器的 x, y, z 坐标
            workspace.append([x, y, z])                             # 保存该末端位置

        self.plotWorkspace(np.array(workspace))                      # 绘制所有采样点构成的工作空间

    def plotWorkspace(self, workspace):
        # X轴
        x_max = workspace[:, 0].max()
        x_min = workspace[:, 0].min()
        x_max_idx = np.argmax(workspace[:, 0])
        x_min_idx = np.argmin(workspace[:, 0])
        x_max_point = workspace[x_max_idx]
        x_min_point = workspace[x_min_idx]

        # Y轴
        y_max = workspace[:, 1].max()
        y_min = workspace[:, 1].min()
        y_max_idx = np.argmax(workspace[:, 1])
        y_min_idx = np.argmin(workspace[:, 1])
        y_max_point = workspace[y_max_idx]
        y_min_point = workspace[y_min_idx]

        # Z轴
        z_max = workspace[:, 2].max()
        z_min = workspace[:, 2].min()
        z_max_idx = np.argmax(workspace[:, 2])
        z_min_idx = np.argmin(workspace[:, 2])
        z_max_point = workspace[z_max_idx]
        z_min_point = workspace[z_min_idx]

        print(f"X轴：最大值 {x_max:.4f} m | 最小值 {x_min:.4f} m")
        print(f"Y轴：最大值 {y_max:.4f} m | 最小值 {y_min:.4f} m")
        print(f"Z轴：最大值 {z_max:.4f} m | 最小值 {z_min:.4f} m")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(workspace[:,0], workspace[:,1], workspace[:,2], s=1, alpha=0.5, c='lightblue', label='workspace')

        ax.scatter(x_max_point[0], x_max_point[1], x_max_point[2],
                s=300, c='red', marker='*', label=f'X Maximum ({x_max:.4f} m)')
        ax.scatter(x_min_point[0], x_min_point[1], x_min_point[2],
                s=300, c='red', marker='d', label=f'X Minimum ({x_min:.4f} m)')

        ax.scatter(y_max_point[0], y_max_point[1], y_max_point[2],
                s=300, c='green', marker='*', label=f'Y Maximum ({y_max:.4f} m)')
        ax.scatter(y_min_point[0], y_min_point[1], y_min_point[2],
                s=300, c='green', marker='d', label=f'Y Minimum ({y_min:.4f} m)')

        ax.scatter(z_max_point[0], z_max_point[1], z_max_point[2],
                s=300, c='blue', marker='*', label=f'Z Maximum ({z_max:.4f} m)')
        ax.scatter(z_min_point[0], z_min_point[1], z_min_point[2],
                s=300, c='blue', marker='d', label=f'Z Minimum ({z_min:.4f} m)')

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('workspace', fontsize=14)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

    def runFunc(self):
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

if __name__ == "__main__":
    robot = GetWorkspace(sample_num=int(5e4))
    robot.runBefore()
