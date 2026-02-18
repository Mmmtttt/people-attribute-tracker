# People Attribute Tracker

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

**基于YOLOv8、DeepSORT和PP-Human的人员属性识别与跟踪系统**

[功能特性](#功能特性) • [快速开始](#快速开始) • [安装指南](#安装指南) • [使用示例](#使用示例) • [结果可视化](#结果可视化)

</div>

## 项目简介

People Attribute Tracker是一个强大的计算机视觉系统，专门用于视频流中的人员检测、跟踪和属性识别。该系统结合了最先进的目标检测、跟踪和属性识别技术，能够准确地识别视频中的人员，并提取包括性别、年龄、朝向、配饰、持物、包、上衣风格和下装风格等8种属性。

### 应用场景

- **零售行业**：店铺门口人流量统计、顾客画像分析
- **安防监控**：人员识别、行为分析
- **智慧城市**：人流统计、人群分析
- **商业智能**：消费者行为分析、市场调研

## 功能特性

### 核心功能

- **人员检测**：使用YOLOv8进行高精度人员检测
- **目标跟踪**：基于DeepSORT算法实现稳定的人员跟踪
- **属性识别**：使用PP-Human模型识别8种人员属性
- **中文显示**：支持中文属性标签显示
- **数据导出**：自动生成CSV和JSON格式的统计数据
- **结果可视化**：提供丰富的统计图表和可视化工具

### 识别属性

| 属性类别 | 识别选项 |
|---------|---------|
| 性别 | 男、女 |
| 年龄段 | 小于18岁、18-60岁、大于60岁 |
| 朝向 | 朝前、朝后、侧面 |
| 配饰 | 眼镜、帽子、无 |
| 正面持物 | 是、否 |
| 包 | 双肩包、单肩包、手提包 |
| 上衣风格 | 带条纹、带logo、带格子、拼接风格 |
| 下装风格 | 带条纹、带图案 |

### 性能指标

- **检测精度**：YOLOv8在COCO数据集上达到53.9% AP
- **跟踪稳定性**：DeepSORT在复杂场景下保持高准确率
- **属性识别准确率**：PP-Human在多个属性上达到90%+准确率
- **处理速度**：
  - 原始版本：4-10 FPS（CPU单线程）
  - 优化版本：6-8 FPS（CPU多线程）
  - 优化版本：15-25 FPS（GPU加速）

### 性能优化

项目提供两个版本：

| 版本 | 文件 | 特点 | 适用场景 |
|-----|------|------|---------|
| 原始版本 | tracker.py | 单线程，CPU | 简单场景，低配置设备 |
| 优化版本 | tracker_optimized.py | GPU加速，多线程 | 高性能需求，实时处理 |

**优化版本特性**：
- ✅ GPU加速（支持NVIDIA CUDA）
- ✅ 多线程处理（自动检测CPU核心数）
- ✅ 性能监控（实时FPS和处理时间统计）
- ✅ 自动配置（智能选择最佳配置）

**性能提升**：
- GPU加速：3-5倍速度提升
- 多线程：20-50%速度提升
- 综合优化：最高可达10倍速度提升

详见：[性能优化文档](docs/PERFORMANCE.md)

## 快速开始

### 前置要求

- Python 3.8 或更高版本
- Windows、Linux 或 macOS 操作系统
- 至少 8GB RAM
- 推荐使用 GPU（可选）

### 一键安装

```bash
# 克隆项目
git clone https://github.com/Mmmtttt/people-attribute-tracker.git
cd people-attribute-tracker

# 安装依赖
pip install -r requirements.txt

# 下载预训练模型（首次运行会自动下载）
python tracker.py --download-models
```

### 快速测试

```bash
# 使用示例视频测试（原始版本）
python tracker.py examples/测试视频.mp4 output/result.mp4

# 使用示例视频测试（优化版本，推荐）
python tracker_optimized.py examples/测试视频.mp4 output/result_optimized.mp4

# 使用GPU加速
python tracker_optimized.py examples/测试视频.mp4 output/result_gpu.mp4 --use-gpu

# 查看结果
# - 输出视频：output/result*.mp4
# - 统计数据：output/*.csv, output/*.json
# - 可视化图表：output/*.png
```

## 安装指南

### 详细安装步骤

#### 1. 环境准备

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

#### 2. 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt

# 或者逐个安装核心依赖
pip install ultralytics
pip install opencv-python
pip install paddlepaddle
pip install deep-sort-realtime
pip install pillow
pip install matplotlib
pip install seaborn
```

#### 3. 下载模型

项目使用以下预训练模型：

- **YOLOv8**：用于人员检测
- **PP-Human**：用于属性识别

模型会自动下载到 `models/` 目录。如果下载失败，可以手动下载：

```bash
# YOLOv8模型（首次运行自动下载）
# PP-Human属性模型
# 下载地址：https://aistudio.baidu.com/projectdetail/4537344
# 解压到 models/PPLCNet_x1_0_person_attribute_945_infer/
```

#### 4. 验证安装

```bash
# 运行测试脚本
python tracker.py --test

# 如果看到 "All tests passed!" 说明安装成功
```

## 使用示例

### 基本用法

#### 原始版本（tracker.py）

```bash
# 处理单个视频
python tracker.py input_video.mp4 output_video.mp4

# 指定置信度阈值
python tracker.py input_video.mp4 output_video.mp4 --conf-threshold 0.6

# 实时显示处理结果
python tracker.py input_video.mp4 output_video.mp4 --display

# 不保存数据文件
python tracker.py input_video.mp4 output_video.mp4 --no-save-data
```

#### 优化版本（tracker_optimized.py）- 推荐

```bash
# 处理单个视频（自动启用GPU和多线程）
python tracker_optimized.py input_video.mp4 output_video.mp4

# 启用GPU加速
python tracker_optimized.py input_video.mp4 output_video.mp4 --use-gpu

# 禁用GPU（使用CPU）
python tracker_optimized.py input_video.mp4 output_video.mp4 --no-gpu

# 启用多线程
python tracker_optimized.py input_video.mp4 output_video.mp4 --use-multithreading

# 指定工作线程数
python tracker_optimized.py input_video.mp4 output_video.mp4 --num-workers 8

# 完整参数
python tracker_optimized.py input_video.mp4 output_video.mp4 \
  --conf-threshold 0.5 \
  --use-gpu \
  --use-multithreading \
  --num-workers 8 \
  --display
```

### 高级用法

```bash
# 使用自定义模型路径
python tracker.py input_video.mp4 output_video.mp4 \
  --model-path custom_yolov8.pt \
  --attribute-model-path custom_attribute_model/

# 批量处理多个视频
python batch_process.py --input-dir input_videos/ --output-dir output/

# 处理摄像头实时视频流
python tracker.py 0 output.mp4 --display

# 使用配置文件
python tracker.py --config config.yaml
```

### Python API

```python
from tracker import YOLOAttributeTracker

# 初始化跟踪器
tracker = YOLOAttributeTracker(
    model_path='yolov8n.pt',
    conf_threshold=0.5,
    attribute_model_path='models/PPLCNet_x1_0_person_attribute_945_infer'
)

# 处理视频
tracker.process_video(
    video_path='input.mp4',
    output_path='output.mp4',
    display=False,
    save_data=True
)

# 访问统计数据
print(f"总人数: {len(tracker.unique_track_ids)}")
print(f"属性识别成功: {len(tracker.track_id_attributes)}")
```

## 结果可视化

项目提供强大的可视化工具，用于分析和展示统计结果。

### 自动生成图表

运行程序后会自动生成以下可视化图表：

- **性别分布饼图**：显示男女比例
- **年龄分布柱状图**：显示各年龄段人数
- **时间序列图**：显示人员数量随时间变化
- **属性热力图**：显示各属性组合的分布

### 使用可视化脚本

```bash
# 生成所有可视化图表
python visualize.py --data output/yolo_attribute_data_20240218_010841.csv

# 指定输出目录
python visualize.py --data output/yolo_attribute_data_20240218_010841.csv --output-dir output/

# 生成特定类型的图表
python visualize.py --data output/yolo_attribute_data_20240218_010841.csv --charts gender age

# 生成交互式图表（需要plotly）
python visualize.py --data output/yolo_attribute_data_20240218_010841.csv --interactive
```

### 图表示例

生成的图表包括：

1. **人员统计概览**
   - 总人数、平均每帧人数、最多同时出现人数
   - 跟踪持续时间分布
   - 人员出现频率分布

2. **属性分析**
   - 性别分布饼图
   - 年龄分布柱状图
   - 各属性组合的交叉分析

3. **时间序列分析**
   - 每帧人员数量变化
   - 累计人员数量增长
   - 高峰时段识别

## 项目结构

```
people-attribute-tracker/
├── tracker.py              # 主程序文件
├── visualize.py            # 可视化脚本
├── batch_process.py        # 批量处理脚本
├── requirements.txt        # 依赖列表
├── README.md               # 项目说明文档
├── LICENSE                 # 许可证
├── config.yaml             # 配置文件示例
├── models/                 # 模型文件目录
│   ├── yolov8n.pt         # YOLOv8模型
│   └── PPLCNet_x1_0_person_attribute_945_infer/  # PP-Human模型
├── examples/               # 示例视频
│   ├── 测试视频.mp4
│   └── 测试视频2.mp4
├── docs/                   # 文档目录
│   ├── INSTALLATION.md     # 详细安装指南
│   ├── API.md             # API文档
│   └── TROUBLESHOOTING.md # 故障排除
└── output/                 # 输出目录
    ├── *.mp4              # 标注视频
    ├── *.csv              # CSV数据文件
    ├── *.json             # JSON统计文件
    └── *.png              # 可视化图表
```

## 配置说明

### 命令行参数

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `video_path` | 输入视频路径 | 必需 |
| `output_path` | 输出视频路径 | None |
| `--conf-threshold` | 检测置信度阈值 | 0.5 |
| `--display` | 实时显示处理结果 | False |
| `--save-data` | 保存数据文件 | True |
| `--attribute-model-path` | 属性识别模型路径 | models/PPLCNet_x1_0_person_attribute_945_infer |
| `--model-path` | YOLOv8模型路径 | yolov8n.pt |

### 配置文件

使用YAML配置文件简化参数设置：

```yaml
# config.yaml
model:
  detection: yolov8n.pt
  attribute: models/PPLCNet_x1_0_person_attribute_945_infer
  conf_threshold: 0.5

tracking:
  max_age: 30
  n_init: 3

output:
  save_video: true
  save_data: true
  save_visualization: true
  output_dir: output/

visualization:
  charts:
    - gender
    - age
    - orientation
    - time_series
```

## 常见问题

### Q: 处理速度太慢怎么办？

A: 可以尝试以下方法：
1. 使用更小的模型（yolov8n.pt）
2. 降低视频分辨率
3. 提高置信度阈值
4. 使用GPU加速

### Q: 中文显示为问号？

A: 确保系统安装了中文字体，程序会自动尝试加载系统字体。

### Q: 属性识别不准确？

A: 属性识别依赖于图像质量，建议：
1. 使用高分辨率视频
2. 确保人员清晰可见
3. 避免过度遮挡

### Q: 如何使用自己的摄像头？

A: 使用摄像头索引号作为输入：
```bash
python tracker.py 0 output.mp4 --display
```

## 技术栈

- **目标检测**：[YOLOv8](https://github.com/ultralytics/ultralytics)
- **目标跟踪**：[DeepSORT](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
- **属性识别**：[PP-Human](https://github.com/PaddlePaddle/PaddleDetection)
- **深度学习框架**：[PaddlePaddle](https://www.paddlepaddle.org.cn/)
- **计算机视觉**：[OpenCV](https://opencv.org/)
- **数据可视化**：[Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- [Ultralytics](https://github.com/ultralytics) - YOLOv8
- [PaddlePaddle](https://github.com/PaddlePaddle) - PP-Human
- [DeepSORT](https://github.com/mikel-brostrom) - 跟踪算法

## 联系方式

- 项目主页：[https://github.com/yourusername/people-attribute-tracker](https://github.com/yourusername/people-attribute-tracker)
- 问题反馈：[Issues](https://github.com/yourusername/people-attribute-tracker/issues)
- 邮箱：your.email@example.com

---

<div align="center">

**如果这个项目对您有帮助，请给个 ⭐️ Star！**

Made with ❤️ by [Your Name]

</div>
