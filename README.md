# CuRobo Realworld

## 环境配置

### 1. 创建虚拟环境

```bash
mkdir curobo_realworld
cd curobo_realworld

pip install uv

uv venv --python 3.10 .venv
source .venv/bin/activate
```

### 2. 安装系统依赖

```bash
sudo apt install git-lfs
```

### 3. 安装 CuRobo 及其依赖

```bash
git clone https://github.com/NVlabs/curobo.git

cd curobo

# 推荐使用 CUDA 12.1:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 如果使用 CUDA 11.8:
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装构建工具
uv pip install setuptools wheel

# 以编辑模式安装 CuRobo (不使用构建隔离以利用已安装的 torch)
uv pip install -e . --no-build-isolation

# 安装测试工具 (可选)
uv pip install pytest

#测试所有模块
python3 -m pytest .
```

### 4. 验证安装

```bash
#使用curobo库自带的Franka.yml测试motion gen
cd ..
python src/Single_plan.py
```

### 5. Piper机械臂

`curobo_realworld` 中的 `piper_camera` 文件夹包含了带相机的 Piper 机械臂相关文件，用于支持 CuRobo 的运动规划。该文件夹内包括：
- **URDF**: 机器人统一描述格式文件 (`piper_description_v100_realsense_camera_v2.urdf`)
- **YML**: CuRobo 机器人配置文件 (`piper.yml、spheres/piper.yml`)

#### 使用方法

> **配置文件路径修改**
> 使用前请务必打开 `piper_camera/piper.yml`，将以下字段的路径修改为您本机的**绝对路径**：
> - `urdf_path`
> - `asset_root_path`
> - `collision_spheres`

#### 示例

```bash
python3 test_curobo/test_piper.py

```