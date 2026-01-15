import numpy as np
import sys
import os

# 添加项目根目录到路径
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.Single_plan import CuroboPlanner

# Piper 初始姿态（6个关节）- 使用 retract_config 作为安全起始点
init_q = np.array([0.0, 1.57, -1.5, 0.0, 0.0, 0.0])

# 目标末端位姿 - 使用正运动学验证过的可达位置
target_pose = {
    "position": [0.38, 0.0, 0.5],  # retract 位置
    "quaternion": [0.713, 0.0, 0.701, 0.0],  # retract 姿态
}

# 移除障碍物进行纯运动学测试
obstacles = None

print("=" * 60)
print("🤖 测试 Piper 机器人 CuRobo 规划")
print("=" * 60)

# 使用本地配置文件的绝对路径
piper_config_path = os.path.join(PROJECT_ROOT, "piper_camera", "piper.yml")
print(f"   配置文件路径: {piper_config_path}")

# 创建规划器
planner = CuroboPlanner(
    robot_cfg_file=piper_config_path,  # 使用 Piper 配置
    obstacles=None,
)

# 执行规划（增加尝试次数和超时时间）
result = planner.plan(
    init_q=init_q,
    target_pose=target_pose,
    max_attempts=20,  # 增加尝试次数
    timeout=10.0,  # 增加超时时间
)

print(f"\n规划成功: {result['success']}")
print(f"状态: {result['status']}")
print(f"耗时: {result['solve_time']:.3f}s")
print(f"轨迹形状: {result['trajectory'].shape}")

#python3 test_curobo/test_piper.py