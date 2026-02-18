# 安装指南

本文档提供了People Attribute Tracker的详细安装步骤。

## 系统要求

### 硬件要求

- **CPU**：Intel i5 或更高，或同等性能的 AMD 处理器
- **内存**：至少 8GB RAM（推荐 16GB）
- **存储**：至少 5GB 可用空间
- **GPU**：可选，NVIDIA GPU（支持 CUDA 11.0+）可显著提升性能

### 软件要求

- **操作系统**：
  - Windows 10/11
  - Linux（Ubuntu 18.04+）
  - macOS 10.15+

- **Python**：3.8 或更高版本（推荐 3.9 或 3.10）

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/people-attribute-tracker.git
cd people-attribute-tracker
```

### 2. 创建虚拟环境（推荐）

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装 Python 依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

如果安装速度较慢，可以使用国内镜像源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 下载预训练模型

#### 方法一：自动下载（推荐）

首次运行程序时，模型会自动下载：

```bash
python tracker.py --download-models
```

#### 方法二：手动下载

如果自动下载失败，可以手动下载模型：

**YOLOv8 模型：**
- 访问：https://github.com/ultralytics/assets/releases
- 下载 `yolov8n.pt`（推荐）或 `yolov8s.pt`
- 放置到项目根目录

**PP-Human 属性识别模型：**
- 访问：https://aistudio.baidu.com/projectdetail/4537344
- 下载 `PPLCNet_x1_0_person_attribute_945_infer`
- 解压到 `models/PPLCNet_x1_0_person_attribute_945_infer/`

### 5. 验证安装

运行测试脚本验证安装是否成功：

```bash
python tracker.py --test
```

如果看到 "All tests passed!"，说明安装成功。

## 常见安装问题

### 问题 1：pip 安装速度慢

**解决方案：**
使用国内镜像源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 2：OpenCV 安装失败

**解决方案：**
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

### 问题 3：PaddlePaddle 安装失败

**解决方案：**

**CPU 版本：**
```bash
pip install paddlepaddle
```

**GPU 版本（CUDA 11.2）：**
```bash
pip install paddlepaddle-gpu
```

**GPU 版本（CUDA 10.2）：**
```bash
pip install paddlepaddle-gpu==2.3.2.post102 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

### 问题 4：模型下载失败

**解决方案：**

1. 检查网络连接
2. 使用代理或 VPN
3. 手动下载模型文件（见上文"手动下载"部分）

### 问题 5：中文显示为问号

**解决方案：**

确保系统安装了中文字体：

**Windows：**
- 系统默认已安装中文字体（如微软雅黑）

**Linux：**
```bash
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei
```

**macOS：**
- 系统默认已安装中文字体

### 问题 6：权限错误

**解决方案：**

**Windows：**
以管理员身份运行命令提示符

**Linux/macOS：**
```bash
sudo pip install -r requirements.txt
```

## 高级配置

### 使用 GPU 加速

如果您有 NVIDIA GPU，可以启用 GPU 加速：

1. 安装 CUDA Toolkit（11.0 或更高版本）
2. 安装 cuDNN
3. 安装 GPU 版本的 PaddlePaddle：

```bash
pip install paddlepaddle-gpu
```

4. 修改 `tracker.py` 中的配置：

```python
config.enable_use_gpu(100, 0)  # 100MB 显存，GPU ID 0
```

### 自定义模型路径

如果您想使用自定义的模型路径，可以修改配置文件 `config.yaml`：

```yaml
model:
  detection: /path/to/your/yolov8.pt
  attribute: /path/to/your/attribute_model/
```

### 调整处理参数

根据您的硬件配置，可以调整以下参数：

- `conf_threshold`：检测置信度阈值（默认 0.5）
- `max_age`：跟踪器最大年龄（默认 30）
- `n_init`：跟踪器初始化帧数（默认 3）

## 卸载

### 完全卸载

```bash
# 停用虚拟环境
deactivate

# 删除虚拟环境
rm -rf venv  # Linux/macOS
# 或
rmdir /s venv  # Windows

# 删除项目目录
rm -rf people-attribute-tracker
```

### 仅卸载依赖

```bash
pip freeze | xargs pip uninstall -y
```

## 获取帮助

如果遇到其他问题：

1. 查看 [README.md](README.md) 中的常见问题部分
2. 搜索 [Issues](https://github.com/yourusername/people-attribute-tracker/issues)
3. 提交新的 Issue，包含：
   - 操作系统版本
   - Python 版本
   - 错误信息
   - 复现步骤

## 下一步

安装完成后，请阅读 [使用指南](USAGE.md) 了解如何使用本系统。
