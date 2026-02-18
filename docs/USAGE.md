# 使用指南

本文档详细介绍了如何使用 People Attribute Tracker 系统。

## 基本使用

### 命令行使用

#### 处理单个视频

```bash
python tracker.py input_video.mp4 output_video.mp4
```

#### 指定置信度阈值

```bash
python tracker.py input_video.mp4 output_video.mp4 --conf-threshold 0.6
```

#### 实时显示处理结果

```bash
python tracker.py input_video.mp4 output_video.mp4 --display
```

#### 不保存数据文件

```bash
python tracker.py input_video.mp4 output_video.mp4 --no-save-data
```

### 使用摄像头

```bash
# 使用默认摄像头（索引 0）
python tracker.py 0 output.mp4 --display

# 使用指定摄像头
python tracker.py 1 output.mp4 --display
```

### 完整参数示例

```bash
python tracker.py input_video.mp4 output_video.mp4 \
  --conf-threshold 0.5 \
  --display \
  --save-data \
  --attribute-model-path models/PPLCNet_x1_0_person_attribute_945_infer \
  --model-path yolov8n.pt
```

## 输出文件说明

运行程序后，会生成以下文件：

### 1. 标注视频

- **文件名**：`output_video.mp4`（或您指定的名称）
- **内容**：包含边界框、跟踪ID和属性标签的视频
- **用途**：直观查看识别结果

### 2. CSV 数据文件

- **文件名**：`yolo_attribute_data_YYYYMMDD_HHMMSS.csv`
- **内容**：每个跟踪人员的详细属性数据
- **格式**：

| 字段 | 说明 |
|-----|------|
| track_id | 跟踪ID |
| gender | 性别（男/女） |
| age | 年龄段（小于18岁/18-60岁/大于60岁） |
| orientation | 朝向（朝前/朝后/侧面） |
| accessory | 配饰（眼镜/帽子/无） |
| hold_object | 正面持物（是/否） |
| bag | 包类型（双肩包/单肩包/手提包） |
| upper_style | 上衣风格（带条纹/带logo/带格子/拼接风格） |
| lower_style | 下装风格（带条纹/带图案） |

### 3. JSON 统计文件

- **文件名**：`yolo_attribute_summary_YYYYMMDD_HHMMSS.json`
- **内容**：完整的统计信息
- **格式**：

```json
{
  "video_file": "input_video.mp4",
  "processing_time": "20240218_010841",
  "total_frames": 900,
  "total_people_detected": 2302,
  "unique_people": 31,
  "average_people_per_frame": 2.66,
  "max_people_in_frame": 7,
  "min_people_in_frame": 0,
  "people_with_attributes": 31,
  "attribute_statistics": {
    "gender": {
      "男": 7,
      "女": 24
    },
    "age": {
      "小于18岁": 4,
      "18-60岁": 25,
      "大于60岁": 2
    }
  }
}
```

## 结果可视化

### 生成所有图表

```bash
python visualize.py --data yolo_attribute_data_20240218_010841.csv --all
```

### 生成特定图表

```bash
# 只生成性别和年龄分布图
python visualize.py --data yolo_attribute_data_20240218_010841.csv --charts gender age

# 生成朝向、配饰和包类型分布图
python visualize.py --data yolo_attribute_data_20240218_010841.csv --charts orientation accessory bag
```

### 指定输出目录

```bash
python visualize.py --data yolo_attribute_data_20240218_010841.csv --output-dir output/
```

### 生成的图表类型

1. **性别分布** (`gender_distribution.png`)
   - 饼图：显示男女比例
   - 柱状图：显示具体人数

2. **年龄分布** (`age_distribution.png`)
   - 柱状图：显示各年龄段人数
   - 饼图：显示年龄段比例

3. **朝向分布** (`orientation_distribution.png`)
   - 柱状图：显示人员朝向统计

4. **配饰分布** (`accessory_distribution.png`)
   - 柱状图：显示配饰类型统计

5. **包类型分布** (`bag_distribution.png`)
   - 柱状图：显示包类型统计

6. **上衣风格分布** (`upper_style_distribution.png`)
   - 柱状图：显示上衣风格统计

7. **下装风格分布** (`lower_style_distribution.png`)
   - 柱状图：显示下装风格统计

8. **统计概览** (`overview.png`)
   - 综合展示所有属性的统计信息

## Python API 使用

### 基本示例

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
print(f"平均每帧人数: {sum(tracker.people_count_per_frame) / len(tracker.people_count_per_frame):.2f}")
```

### 高级示例

```python
from tracker import YOLOAttributeTracker
import cv2

# 初始化跟踪器
tracker = YOLOAttributeTracker(
    model_path='yolov8n.pt',
    conf_threshold=0.6,
    attribute_model_path='models/PPLCNet_x1_0_person_attribute_945_infer'
)

# 打开视频文件
cap = cv2.VideoCapture('input.mp4')

# 逐帧处理
frame_number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 处理单帧
    processed_frame, current_people, total_unique = tracker.process_frame(frame, frame_number)
    
    # 显示结果
    cv2.imshow('Tracking', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_number += 1

cap.release()
cv2.destroyAllWindows()

# 保存数据
tracker.save_data('input.mp4', 30)
```

### 访问属性数据

```python
from tracker import YOLOAttributeTracker

# 初始化并处理视频
tracker = YOLOAttributeTracker()
tracker.process_video('input.mp4', 'output.mp4', display=False)

# 访问每个人员的属性
for track_id, attrs in tracker.track_id_attributes.items():
    print(f"Track ID: {track_id}")
    parsed = attrs['parsed_attributes']
    print(f"  性别: {parsed.get('性别', '?')}")
    print(f"  年龄: {parsed.get('年龄段', '?')}")
    print(f"  朝向: {parsed.get('朝向', '?')}")
    print(f"  配饰: {parsed.get('配饰', '?')}")
    print(f"  持物: {parsed.get('正面持物', '?')}")
    print(f"  包: {parsed.get('包', '?')}")
    print(f"  上衣: {parsed.get('上衣风格', '?')}")
    print(f"  下装: {parsed.get('下装风格', '?')}")
    print()
```

## 批量处理

### 批量处理多个视频

创建 `batch_process.py`：

```python
import os
import sys
from tracker import YOLOAttributeTracker

def batch_process(input_dir, output_dir, conf_threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化跟踪器
    tracker = YOLOAttributeTracker(conf_threshold=conf_threshold)
    
    # 获取所有视频文件
    video_files = [f for f in os.listdir(input_dir) 
                   if f.endswith(('.mp4', '.avi', '.mov'))]
    
    print(f"Found {len(video_files)} video files")
    
    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, f"processed_{video_file}")
        
        print(f"\nProcessing: {video_file}")
        tracker.process_video(input_path, output_path, display=False, save_data=True)
        print(f"Completed: {output_path}")

if __name__ == '__main__':
    input_dir = sys.argv[1] if len(sys.argv) > 1 else 'input_videos/'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'output/'
    conf_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    batch_process(input_dir, output_dir, conf_threshold)
```

运行批量处理：

```bash
python batch_process.py input_videos/ output/ 0.5
```

## 性能优化

### 提高处理速度

1. **使用更小的模型**

```bash
python tracker.py input.mp4 output.mp4 --model-path yolov8n.pt
```

2. **提高置信度阈值**

```bash
python tracker.py input.mp4 output.mp4 --conf-threshold 0.7
```

3. **降低视频分辨率**

使用 FFmpeg 预处理视频：

```bash
ffmpeg -i input.mp4 -vf scale=640:360 input_small.mp4
python tracker.py input_small.mp4 output.mp4
```

4. **使用 GPU 加速**

确保安装了 GPU 版本的 PaddlePaddle：

```bash
pip install paddlepaddle-gpu
```

### 提高识别准确率

1. **使用更大的模型**

```bash
python tracker.py input.mp4 output.mp4 --model-path yolov8s.pt
```

2. **降低置信度阈值**

```bash
python tracker.py input.mp4 output.mp4 --conf-threshold 0.3
```

3. **使用高分辨率视频**

确保输入视频分辨率至少为 720p。

## 常见使用场景

### 场景 1：店铺门口人流量统计

```bash
python tracker.py shop_entrance.mp4 shop_output.mp4 --conf-threshold 0.5
```

### 场景 2：商场顾客画像分析

```bash
python tracker.py mall_footage.mp4 mall_output.mp4 --save-data
python visualize.py --data yolo_attribute_data_*.csv --all
```

### 场景 3：实时监控分析

```bash
python tracker.py 0 real_time_output.mp4 --display
```

### 场景 4：历史视频批量分析

```bash
python batch_process.py historical_videos/ analysis_results/
```

## 故障排除

### 问题：处理速度太慢

**解决方案：**
1. 使用更小的模型（yolov8n.pt）
2. 提高置信度阈值
3. 降低视频分辨率
4. 使用 GPU 加速

### 问题：识别准确率低

**解决方案：**
1. 使用更大的模型（yolov8s.pt 或 yolov8m.pt）
2. 降低置信度阈值
3. 使用高分辨率视频
4. 确保视频质量良好

### 问题：跟踪不稳定

**解决方案：**
1. 调整跟踪器参数（max_age, n_init）
2. 使用更高的帧率视频
3. 确保人员移动速度适中

### 问题：属性识别不准确

**解决方案：**
1. 确保人员清晰可见
2. 避免过度遮挡
3. 使用高分辨率视频
4. 确保光照条件良好

## 下一步

- 查看 [API 文档](API.md) 了解更多 API 细节
- 查看 [故障排除](TROUBLESHOOTING.md) 解决常见问题
- 查看 [README.md](../README.md) 了解项目概览
