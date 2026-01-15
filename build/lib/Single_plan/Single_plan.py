
import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import torch

print("✅ 基础库导入成功")

import curobo
print("✅ 成功导入 curobo")

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.geom.types import Cuboid, Mesh, Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

print("✅ CuRobo 导入成功")


class CuroboPlanner:
    """
    CuRobo 轨迹规划器封装类
    
    初始化一次后可多次调用 plan() 方法进行规划
    """
    
    def __init__(
        self,
        robot_cfg_file: str = "franka.yml",
        obstacles: Optional[Union[Dict, List, WorldConfig]] = None,
        interpolation_dt: float = 0.02,
        use_cuda_graph: bool = True,
        collision_checker_type: CollisionCheckerType = CollisionCheckerType.MESH,
        collision_activation_distance: float = 0.02,
        smooth_weight: Optional[List[float]] = None,
        velocity_scale: Optional[Union[List[float], float]] = None,
        acceleration_scale: Optional[Union[List[float], float]] = None,
    ):
        """
        初始化 CuRobo 规划器
        
        Args:
            robot_cfg_file: 机器人配置文件（curobo 自带 franka.yml, ur5e.yml 等）
            obstacles: 障碍物配置
            interpolation_dt: 插值时间步长（秒）
            use_cuda_graph: 是否使用 CUDA Graph 加速
            collision_checker_type: 碰撞检测类型
        """
        self.tensor_args = TensorDeviceType()
        self.interpolation_dt = interpolation_dt
        
        # 解析障碍物配置
        world_config = self._parse_obstacles(obstacles)
        
        # 加载 MotionGen 配置
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg_file,
            world_config,
            self.tensor_args,
            interpolation_dt=interpolation_dt,
            collision_checker_type=collision_checker_type,
            use_cuda_graph=use_cuda_graph,
            trajopt_tsteps=32,  # 轨迹优化时间步数
            interpolation_steps=5000,  # 插值缓冲区大小
            collision_activation_distance=collision_activation_distance,
            smooth_weight=smooth_weight,
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
        )
        
        # 创建 MotionGen 实例
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup(parallel_finetune=True)
        
        # 获取关节名称
        self.joint_names = self.motion_gen.kinematics.joint_names
        
        print(f"✅ CuroboPlanner 初始化完成")
        print(f"   关节名称: {self.joint_names}")
    
    def _parse_obstacles(
        self, obstacles: Optional[Union[Dict, List, WorldConfig]]
    ) -> Optional[WorldConfig]:
        """解析障碍物配置为 WorldConfig"""
        if obstacles is None:
            return None
        
        if isinstance(obstacles, WorldConfig):
            return obstacles
        
        if isinstance(obstacles, dict):
            return WorldConfig.from_dict(obstacles)
        
        if isinstance(obstacles, list):
            cuboids = [o for o in obstacles if isinstance(o, Cuboid)]
            spheres = [o for o in obstacles if isinstance(o, Sphere)]
            meshes = [o for o in obstacles if isinstance(o, Mesh)]
            return WorldConfig(cuboid=cuboids, sphere=spheres, mesh=meshes)
        
        raise ValueError(f"不支持的障碍物类型: {type(obstacles)}")
    
    def update_world(self, obstacles: Union[Dict, List, WorldConfig]):
        """更新障碍物（在场景变化时调用）"""
        world_config = self._parse_obstacles(obstacles)
        self.motion_gen.update_world(world_config)
    
    def plan(
        self,
        init_q: np.ndarray,
        target_pose: Dict,
        init_qd: Optional[np.ndarray] = None,
        max_attempts: int = 10,
        timeout: float = 5.0,
        time_dilation_factor: float = 1.0,
        enable_graph: bool = True,
        pose_cost_metric: Optional[PoseCostMetric] = None,
        rotation_weight: Optional[float] = None,
    ) -> Dict:
        """
        规划从当前关节状态到目标末端位姿的轨迹
        
        Args:
            init_q: 初始关节角度 (7,)
            target_pose: 目标位姿 {"position": [x,y,z], "quaternion": [w,x,y,z]}
            init_qd: 初始关节速度 (7,)，默认为 0
            max_attempts: 最大尝试次数
            timeout: 超时时间（秒）
            time_dilation_factor: 时间缩放因子（<1 会生成更慢的轨迹）
            enable_graph: 是否启用图搜索（失败时回退）
            pose_cost_metric: 自定义姿态代价权重 (PoseCostMetric)
            rotation_weight: 旋转权重简便设置 (0.0 表示忽略姿态，1.0 表示正常)

        示例：
            result = planner.plan(
            init_q=init_q,
            target_pose=TARGET_POSE,
            max_attempts=50,
            timeout=10.0,
            rotation_weight=0.01,  # 允许较大的姿态误差
        )
        
        Returns:
            dict: 包含 success, trajectory, dt, status, solve_time 等
        """
        # 构建初始关节状态
        init_q_tensor = self.tensor_args.to_device(init_q).view(1, -1)
        if init_qd is None:
            init_qd = np.zeros_like(init_q)
        init_qd_tensor = self.tensor_args.to_device(init_qd).view(1, -1)
        
        start_state = JointState(
            position=init_q_tensor,
            velocity=init_qd_tensor * 0.0,  # 静止启动更稳定
            acceleration=init_qd_tensor * 0.0,
            jerk=init_qd_tensor * 0.0,
            joint_names=self.joint_names,
        )
        
        # 构建目标位姿（注意 CuRobo 的四元数是 wxyz 顺序）
        goal_position = self.tensor_args.to_device(target_pose["position"])
        goal_quaternion = self.tensor_args.to_device(target_pose["quaternion"])
        goal_pose = Pose(position=goal_position, quaternion=goal_quaternion)
        
        if pose_cost_metric is None:
            if rotation_weight is not None:
                # 如果用户指定了旋转权重，创建一个新的 PoseCostMetric
                # CuRobo 中权重顺序通常为 [rx, ry, rz, x, y, z]
                # 设置 reach_partial_pose=True 并提供 reach_vec_weight
                weight_vec = torch.tensor(
                    [rotation_weight, rotation_weight, rotation_weight, 1.0, 1.0, 1.0], 
                    device=self.tensor_args.device
                )
                pose_cost_metric = PoseCostMetric(
                    reach_partial_pose=True,
                    reach_vec_weight=weight_vec
                )
        
        # 规划配置
        plan_config = MotionGenPlanConfig(
            max_attempts=max_attempts,
            timeout=timeout,
            time_dilation_factor=time_dilation_factor,
            enable_graph=enable_graph,
            enable_finetune_trajopt=True,
            parallel_finetune=True,
            pose_cost_metric=pose_cost_metric,
        )
        
        # 执行规划
        result = self.motion_gen.plan_single(start_state, goal_pose, plan_config)
        
        # 解析结果
        success = result.success.item()
        status = str(result.status)
        solve_time = result.solve_time
        
        if success:
            # 获取插值后的轨迹
            traj = result.get_interpolated_plan()
            trajectory = traj.position.cpu().numpy()
            dt = result.interpolation_dt
            position_error = result.position_error.item() if result.position_error is not None else 0.0
            rotation_error = result.rotation_error.item() if result.rotation_error is not None else 0.0
        else:
            trajectory = np.zeros((1, len(self.joint_names)))
            dt = self.interpolation_dt
            position_error = float("inf")
            rotation_error = float("inf")
        
        return {
            "success": success,
            "trajectory": trajectory,
            "dt": dt,
            "status": status,
            "solve_time": solve_time,
            "position_error": position_error,
            "rotation_error": rotation_error,
        }


def plan_trajectory(
    init_q: np.ndarray,
    target_pose: Dict,
    robot_cfg_file: str = "franka.yml",
    obstacles: Optional[Union[Dict, List]] = None,
    interpolation_dt: float = 0.02,
    max_attempts: int = 10,
    timeout: float = 5.0,
) -> Dict:
    """
    一次性规划函数（会重新初始化 MotionGen
    
    Args:
        init_q: 初始关节角度 (7,)
        target_pose: 目标位姿 {"position": [x,y,z], "quaternion": [w,x,y,z]}
        robot_cfg_file: 机器人配置文件
        obstacles: 障碍物配置
        interpolation_dt: 插值时间步长
        max_attempts: 最大尝试次数
        timeout: 超时时间
    
    Returns:
        dict: 包含 success, trajectory, dt, status 等
    """
    planner = CuroboPlanner(
        robot_cfg_file=robot_cfg_file,
        obstacles=obstacles,
        interpolation_dt=interpolation_dt,
    )
    return planner.plan(
        init_q=init_q,
        target_pose=target_pose,
        max_attempts=max_attempts,
        timeout=timeout,
    )


# ============ 示例 / 测试代码 ============
if __name__ == "__main__":
    print("📦 导入完成，准备初始化...")
    
    # 初始关节角度（Franka home pose）
    init_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    
    # 目标末端位姿（相对于机器人 base）
    target_pose = {
        "position": [0.4, 0.0, 0.4],  # x, y, z (米)
        "quaternion": [0.0, 1.0, 0.0, 0.0],  # w, x, y, z (末端朝下)
    }
    
    obstacles = {
        "cuboid": {
            "Cube": {
                "dims": [1.0, 1.0, 0.1],  # x, y, z 尺寸（米）
                "pose": [0.5, 0.0, -0.05, 1, 0, 0, 0],  # x, y, z, qw, qx, qy, qz
            },
        },
    }
    
    # 方式1: 使用一次性函数
    print("=" * 50)
    print("方式1: 使用 plan_trajectory() 一次性函数")
    print("=" * 50)
    result = plan_trajectory(
        init_q=init_q,
        target_pose=target_pose,
        obstacles=obstacles,
    )
    print(f"规划成功: {result['success']}")
    print(f"状态: {result['status']}")
    print(f"耗时: {result['solve_time']:.3f}s")
    print(f"轨迹形状: {result['trajectory'].shape}")
    print(f"时间步长: {result['dt']:.4f}s")
    
    #python3 Single_plan.py