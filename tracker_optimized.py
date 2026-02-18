from ultralytics import YOLO
import cv2
import time
import os
import json
import csv
import numpy as np
import argparse
from datetime import datetime
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
import paddle
from PIL import Image, ImageDraw, ImageFont
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

class PerformanceMonitor:
    def __init__(self):
        self.fps_history = []
        self.frame_times = defaultdict(list)
        self.total_frames = 0
        self.start_time = None
        
    def start(self):
        self.start_time = time.time()
        
    def record_frame(self, frame_type, duration):
        self.frame_times[frame_type].append(duration)
        
    def calculate_fps(self):
        if self.start_time and self.total_frames > 0:
            elapsed = time.time() - self.start_time
            fps = self.total_frames / elapsed
            self.fps_history.append(fps)
            return fps
        return 0
    
    def get_average_fps(self):
        if self.fps_history:
            return np.mean(self.fps_history)
        return 0
    
    def get_stats(self):
        stats = {}
        for frame_type, times in self.frame_times.items():
            if times:
                stats[frame_type] = {
                    'avg': np.mean(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times)
                }
        return stats

class YOLOAttributeTrackerOptimized:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5, attribute_model_path=None, 
                 use_gpu=True, use_multithreading=True, num_workers=None):
        print("YOLOv8 + DeepSORT + PP-Human Attribute Tracking System (Optimized)")
        print("="*60)
        
        self.use_gpu = use_gpu
        self.use_multithreading = use_multithreading
        self.num_workers = num_workers or (mp.cpu_count() if use_multithreading else 1)
        
        print(f"Performance Configuration:")
        print(f"  GPU Acceleration: {'Enabled' if use_gpu else 'Disabled'}")
        print(f"  Multi-threading: {'Enabled' if use_multithreading else 'Disabled'}")
        print(f"  Number of Workers: {self.num_workers}")
        print("="*60)
        
        self.performance_monitor = PerformanceMonitor()
        
        print(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        print("Initializing DeepSORT tracker...")
        self.tracker = DeepSort(max_age=30, n_init=3)
        
        if attribute_model_path is None:
            attribute_model_path = 'models/models/PPLCNet_x1_0_person_attribute_945_infer'
        
        if attribute_model_path:
            print(f"Loading PP-Human attribute model from {attribute_model_path}...")
            self.attribute_predictor, self.input_name, self.output_name = self.load_model(attribute_model_path)
        else:
            self.attribute_predictor = None
            print("Attribute recognition disabled")
        
        print("="*60)
        
        self.people_count_per_frame = []
        self.total_people_detected = 0
        self.unique_track_ids = set()
        self.track_id_count = defaultdict(int)
        self.frame_data = []
        self.track_id_attributes = {}
        self.track_history = {}
        self.track_timeout = 30
        
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
    
    def check_gpu_available(self):
        try:
            if paddle.is_compiled_with_cuda():
                gpu_count = paddle.device.cuda.device_count()
                if gpu_count > 0:
                    print(f"GPU detected: {gpu_count} device(s) available")
                    return True
        except Exception as e:
            print(f"GPU check failed: {e}")
        return False
    
    def load_model(self, model_dir):
        try:
            model_file = os.path.join(model_dir, 'inference.pdmodel')
            params_file = os.path.join(model_dir, 'inference.pdiparams')
            
            if not os.path.exists(model_file) or not os.path.exists(params_file):
                print(f"Warning: Model files not found in {model_dir}")
                return None, None, None
            
            config = paddle.inference.Config(model_file, params_file)
            
            if self.use_gpu and self.check_gpu_available():
                print("Enabling GPU acceleration for PP-Human model...")
                config.enable_use_gpu(100, 0)
                config.enable_memory_optim()
            else:
                print("Using CPU for PP-Human model...")
                config.disable_gpu()
                config.set_cpu_math_library_num_threads(self.num_workers)
            
            config.switch_ir_optim(False)
            
            predictor = paddle.inference.create_predictor(config)
            
            input_names = predictor.get_input_names()
            output_names = predictor.get_output_names()
            
            input_name = input_names[0] if input_names else 'x'
            output_name = output_names[0] if output_names else 'output'
            
            print(f"PP-Human attribute model loaded successfully from {model_dir}!")
            print(f"Input name: {input_name}, Output name: {output_name}")
            return predictor, input_name, output_name
        except Exception as e:
            print(f"Error loading PP-Human attribute model: {e}")
            return None, None, None
    
    def draw_chinese_text(self, frame, text, position, font_size=20, color=(0, 255, 0)):
        try:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            try:
                font = ImageFont.truetype("msyh.ttc", font_size)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            rgb_color = (color[2], color[1], color[0])
            draw.text(position, text, font=font, fill=rgb_color)
            
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return frame
        except Exception as e:
            print(f"Error drawing Chinese text: {e}")
            return frame
    
    def process_frame(self, frame, frame_number):
        start_time = time.time()
        
        results = self.model(frame, classes=[0], conf=self.conf_threshold, verbose=False)
        detection_time = time.time() - start_time
        self.performance_monitor.record_frame('detection', detection_time)
        
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            
            detections.append([
                [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                float(confidence),
                'person'
            ])
        
        track_start = time.time()
        tracks = self.tracker.update_tracks(detections, frame=frame)
        track_time = time.time() - track_start
        self.performance_monitor.record_frame('tracking', track_time)
        
        people_count = len(tracks)
        self.people_count_per_frame.append(people_count)
        
        current_frame_data = {
            'frame': frame_number,
            'timestamp': frame_number / 29.0,
            'people_count': people_count,
            'track_ids': []
        }
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            self.unique_track_ids.add(track_id)
            self.track_id_count[track_id] += 1
            self.total_people_detected += 1
            
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            color = self.get_color(track_id)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID:{track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if self.attribute_predictor and track_id not in self.track_id_attributes:
                person_img = frame[max(0, y1-10):min(frame.shape[0], y2+10), 
                                  max(0, x1-10):min(frame.shape[1], x2+10)]
                
                if person_img.size > 0:
                    attr_start = time.time()
                    attributes = self.recognize_attributes(person_img)
                    attr_time = time.time() - attr_start
                    self.performance_monitor.record_frame('attribute_recognition', attr_time)
                    
                    parsed_attributes = self.parse_attributes(attributes)
                    self.track_id_attributes[track_id] = {
                        'raw_attributes': attributes,
                        'parsed_attributes': parsed_attributes
                    }
            
            if track_id in self.track_id_attributes:
                attrs = self.track_id_attributes[track_id]['parsed_attributes']
                
                info_lines = []
                gender = attrs.get('性别', '?')
                age = attrs.get('年龄段', '?')
                orientation = attrs.get('朝向', '?')
                accessory = attrs.get('配饰', '?')
                hold_object = attrs.get('正面持物', '?')
                bag = attrs.get('包', '?')
                upper = attrs.get('上衣风格', '?')
                lower = attrs.get('下装风格', '?')
                
                info_lines.append(f"性别: {gender}")
                info_lines.append(f"年龄: {age}")
                info_lines.append(f"朝向: {orientation}")
                info_lines.append(f"配饰: {accessory}")
                info_lines.append(f"持物: {hold_object}")
                info_lines.append(f"包: {bag}")
                info_lines.append(f"上衣: {upper}")
                info_lines.append(f"下装: {lower}")
                
                for i, line in enumerate(info_lines):
                    y_offset = y1 - 30 - (i * 20)
                    if y_offset > 10:
                        frame = self.draw_chinese_text(frame, line, (x1, y_offset), 
                                                  font_size=16, color=color)
            
            current_frame_data['track_ids'].append({
                'id': track_id,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })
        
        self.frame_data.append(current_frame_data)
        
        total_time = time.time() - start_time
        self.performance_monitor.record_frame('total', total_time)
        
        return frame, people_count, len(self.unique_track_ids)
    
    def recognize_attributes(self, image):
        if not self.attribute_predictor:
            return {}
        
        try:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image = cv2.resize(image, (256, 256))
            image = image.astype('float32') / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, axis=0)
            
            input_handle = self.attribute_predictor.get_input_handle(self.input_name)
            input_handle.copy_from_cpu(image)
            self.attribute_predictor.run()
            
            output_handle = self.attribute_predictor.get_output_handle(self.output_name)
            output = output_handle.copy_to_cpu()
            
            attributes = {}
            for i, score in enumerate(output[0]):
                attributes[f'attr_{i}'] = float(score)
            
            return attributes
            
        except Exception as e:
            print(f"Error recognizing attributes: {e}")
            return {}
    
    def parse_attributes(self, attributes):
        if not attributes:
            return {}
        
        parsed = {}
        attr_keys = list(attributes.keys())
        
        if len(attr_keys) >= 2:
            gender_scores = [('男', attributes.get(attr_keys[0], 0)), ('女', attributes.get(attr_keys[1], 0))]
            parsed['性别'] = max(gender_scores, key=lambda x: x[1])[0]
        
        if len(attr_keys) >= 5:
            age_scores = [('小于18岁', attributes.get(attr_keys[2], 0)), 
                         ('18-60岁', attributes.get(attr_keys[3], 0)), 
                         ('大于60岁', attributes.get(attr_keys[4], 0))]
            parsed['年龄段'] = max(age_scores, key=lambda x: x[1])[0]
        
        if len(attr_keys) >= 8:
            orientation_scores = [('朝前', attributes.get(attr_keys[5], 0)), 
                                ('朝后', attributes.get(attr_keys[6], 0)), 
                                ('侧面', attributes.get(attr_keys[7], 0))]
            parsed['朝向'] = max(orientation_scores, key=lambda x: x[1])[0]
        
        if len(attr_keys) >= 11:
            accessory_scores = [('眼镜', attributes.get(attr_keys[8], 0)), 
                              ('帽子', attributes.get(attr_keys[9], 0)), 
                              ('无', attributes.get(attr_keys[10], 0))]
            parsed['配饰'] = max(accessory_scores, key=lambda x: x[1])[0]
        
        if len(attr_keys) >= 13:
            parsed['正面持物'] = '是' if attributes.get(attr_keys[11], 0) > attributes.get(attr_keys[12], 0) else '否'
        
        if len(attr_keys) >= 16:
            bag_scores = [('双肩包', attributes.get(attr_keys[13], 0)), 
                         ('单肩包', attributes.get(attr_keys[14], 0)), 
                         ('手提包', attributes.get(attr_keys[15], 0))]
            parsed['包'] = max(bag_scores, key=lambda x: x[1])[0]
        
        if len(attr_keys) >= 20:
            upper_scores = [('带条纹', attributes.get(attr_keys[16], 0)), 
                           ('带logo', attributes.get(attr_keys[17], 0)), 
                           ('带格子', attributes.get(attr_keys[18], 0)), 
                           ('拼接风格', attributes.get(attr_keys[19], 0))]
            parsed['上衣风格'] = max(upper_scores, key=lambda x: x[1])[0]
        
        if len(attr_keys) >= 22:
            lower_scores = [('带条纹', attributes.get(attr_keys[20], 0)), 
                           ('带图案', attributes.get(attr_keys[21], 0))]
            parsed['下装风格'] = max(lower_scores, key=lambda x: x[1])[0]
        
        return parsed
    
    def get_color(self, track_id):
        colors = [
            (0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255),
            (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128),
            (255, 255, 0), (128, 255, 0), (255, 0, 128), (0, 128, 128)
        ]
        return colors[int(track_id) % len(colors)]
    
    def process_video(self, video_path, output_path=None, display=True, save_data=True):
        self.performance_monitor.start()
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video file: {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*60}")
        print(f"Video Information:")
        print(f"{'='*60}")
        print(f"File: {video_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total Frames: {total_frames}")
        print(f"Duration: {total_frames/fps:.2f} seconds")
        print(f"{'='*60}\n")
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, current_people, total_unique = self.process_frame(frame, frame_count)
            
            info_text = f"People: {current_people} | Unique: {total_unique} | Frame: {frame_count}/{total_frames} | FPS: {self.performance_monitor.calculate_fps():.2f}"
            cv2.putText(processed_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if output_path:
                out.write(processed_frame)
            
            if display:
                cv2.imshow('YOLO Attribute Tracking (Optimized)', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            self.performance_monitor.total_frames = frame_count
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_real = frame_count / elapsed
                print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%), "
                      f"Real-time FPS: {fps_real:.2f}, Unique people: {total_unique}")
        
        cap.release()
        if output_path:
            out.release()
        if display:
            cv2.destroyAllWindows()
        
        self.print_performance_stats()
        self.print_statistics(frame_count, total_frames, fps)
        
        if save_data:
            self.save_data(video_path, fps)
    
    def print_performance_stats(self):
        print(f"\n{'='*60}")
        print(f"Performance Statistics:")
        print(f"{'='*60}")
        
        avg_fps = self.performance_monitor.get_average_fps()
        print(f"Average FPS: {avg_fps:.2f}")
        
        stats = self.performance_monitor.get_stats()
        for frame_type, type_stats in stats.items():
            print(f"\n{frame_type}:")
            print(f"  Average time: {type_stats['avg']*1000:.2f} ms")
            print(f"  Min time: {type_stats['min']*1000:.2f} ms")
            print(f"  Max time: {type_stats['max']*1000:.2f} ms")
            print(f"  Count: {type_stats['count']}")
    
    def save_data(self, video_path, fps):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_filename = f"yolo_attribute_data_{timestamp}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['track_id', 'gender', 'age', 'orientation', 'accessory', 'hold_object', 'bag', 'upper_style', 'lower_style']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for track_id in self.unique_track_ids:
                row = {'track_id': track_id}
                if track_id in self.track_id_attributes:
                    attrs = self.track_id_attributes[track_id]['parsed_attributes']
                    row['gender'] = attrs.get('性别', '?')
                    row['age'] = attrs.get('年 龄段', attrs.get('年龄段', '?'))
                    row['orientation'] = attrs.get('朝向', '?')
                    row['accessory'] = attrs.get('配饰', '?')
                    row['hold_object'] = attrs.get('正面 持物', attrs.get('正面持物', '?'))
                    row['bag'] = attrs.get('包', '?')
                    row['upper_style'] = attrs.get('上衣风格', '?')
                    row['lower_style'] = attrs.get('下装风格', '?')
                writer.writerow(row)
        
        json_filename = f"yolo_attribute_summary_{timestamp}.json"
        summary = {
            'video_file': video_path,
            'processing_time': timestamp,
            'total_frames': len(self.frame_data),
            'total_people_detected': self.total_people_detected,
            'unique_people': len(self.unique_track_ids),
            'average_people_per_frame': sum(self.people_count_per_frame) / len(self.people_count_per_frame) if self.people_count_per_frame else 0,
            'max_people_in_frame': max(self.people_count_per_frame) if self.people_count_per_frame else 0,
            'min_people_in_frame': min(self.people_count_per_frame) if self.people_count_per_frame else 0,
            'people_with_attributes': len(self.track_id_attributes),
            'performance_stats': {
                'average_fps': self.performance_monitor.get_average_fps(),
                'gpu_enabled': self.use_gpu,
                'multithreading_enabled': self.use_multithreading,
                'num_workers': self.num_workers
            },
            'attribute_statistics': {
                'gender': {},
                'age': {}
            }
        }
        
        if self.track_id_attributes:
            gender_stats = defaultdict(int)
            age_stats = defaultdict(int)
            
            for attrs in self.track_id_attributes.values():
                parsed = attrs.get('parsed_attributes', {})
                gender = parsed.get('性别', '未知')
                age = parsed.get('年龄段', '未知')
                gender_stats[gender] += 1
                age_stats[age] += 1
            
            summary['attribute_statistics']['gender'] = dict(gender_stats)
            summary['attribute_statistics']['age'] = dict(age_stats)
        
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(summary, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"\nData saved:")
        print(f"  - CSV: {csv_filename}")
        print(f"  - JSON: {json_filename}")
    
    def print_statistics(self, processed_frames, total_frames, fps):
        print(f"\n{'='*60}")
        print(f"Tracking Statistics:")
        print(f"{'='*60}")
        print(f"Total frames processed: {processed_frames}/{total_frames}")
        print(f"Total unique people tracked: {len(self.unique_track_ids)}")
        print(f"People with attributes: {len(self.track_id_attributes)}")
        
        if self.people_count_per_frame:
            avg_people = sum(self.people_count_per_frame) / len(self.people_count_per_frame)
            max_people = max(self.people_count_per_frame)
            min_people = min(self.people_count_per_frame)
            
            print(f"Average people per frame: {avg_people:.2f}")
            print(f"Maximum people in a frame: {max_people}")
            print(f"Minimum people in a frame: {min_people}")
            
            people_distribution = defaultdict(int)
            for count in self.people_count_per_frame:
                people_distribution[count] += 1
            
            print(f"\nPeople Distribution:")
            for count in sorted(people_distribution.keys()):
                percentage = (people_distribution[count] / len(self.people_count_per_frame)) * 100
                print(f"  {count} people: {people_distribution[count]} frames ({percentage:.1f}%)")
        
        print(f"\nTrack ID Statistics:")
        print(f"Total track detections: {self.total_people_detected}")
        print(f"Average frames per person: {self.total_people_detected / len(self.unique_track_ids):.2f}")
        
        print(f"\nTrack Duration Distribution:")
        duration_distribution = defaultdict(int)
        for track_id, count in self.track_id_count.items():
            duration = count // fps
            duration_distribution[duration] += 1
        
        for duration in sorted(duration_distribution.keys()):
            count = duration_distribution[duration]
            percentage = (count / len(self.track_id_count)) * 100
            print(f"  {duration} seconds: {count} people ({percentage:.1f}%)")
        
        if self.track_id_attributes:
            print(f"\nAttribute Statistics:")
            gender_stats = defaultdict(int)
            age_stats = defaultdict(int)
            
            for attrs in self.track_id_attributes.values():
                parsed = attrs.get('parsed_attributes', {})
                gender = parsed.get('性别', '未知')
                age = parsed.get('年龄段', '未知')
                gender_stats[gender] += 1
                age_stats[age] += 1
            
            print(f"\nGender Distribution:")
            for gender, count in gender_stats.items():
                percentage = (count / len(self.track_id_attributes)) * 100
                print(f"  {gender}: {count} people ({percentage:.1f}%)")
            
            print(f"\nAge Distribution:")
            for age, count in age_stats.items():
                percentage = (count / len(self.track_id_attributes)) * 100
                print(f"  {age}: {count} people ({percentage:.1f}%)")
        
        print(f"{'='*60}\n")

def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='People Attribute Tracker (Optimized) - YOLOv8 + DeepSORT + PP-Human')
    parser.add_argument('video_path', type=str, nargs='?', help='Input video path or camera index')
    parser.add_argument('output_path', type=str, nargs='?', default=None, help='Output video path')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--display', action='store_true', help='Display video in real-time')
    parser.add_argument('--no-save-data', action='store_true', help='Do not save data files')
    parser.add_argument('--attribute-model-path', type=str, default='models/models/PPLCNet_x1_0_person_attribute_945_infer', 
                       help='Path to PP-Human attribute model')
    parser.add_argument('--model-path', type=str, default='yolov8n.pt', help='Path to YOLOv8 model')
    parser.add_argument('--test', action='store_true', help='Run test and exit')
    parser.add_argument('--download-models', action='store_true', help='Download pre-trained models')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='Enable GPU acceleration')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--use-multithreading', action='store_true', default=True, help='Enable multi-threading')
    parser.add_argument('--no-multithreading', action='store_true', help='Disable multi-threading')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of worker threads')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running tests...")
        try:
            import cv2
            import paddle
            from ultralytics import YOLO
            print("All dependencies installed successfully!")
            print("Test passed!")
            return
        except ImportError as e:
            print(f"Test failed: {e}")
            return
    
    if args.download_models:
        print("Downloading models...")
        try:
            from ultralytics import YOLO
            print("Downloading YOLOv8 model...")
            model = YOLO(args.model_path)
            print(f"YOLOv8 model downloaded to: {args.model_path}")
            print("Note: PP-Human model needs to be downloaded manually from:")
            print("https://aistudio.baidu.com/projectdetail/4537344")
        except Exception as e:
            print(f"Error downloading models: {e}")
        return
    
    if not args.video_path:
        parser.print_help()
        return
    
    use_gpu = args.use_gpu and not args.no_gpu
    use_multithreading = args.use_multithreading and not args.no_multithreading
    num_workers = args.num_workers
    
    save_data = not args.no_save_data
    
    tracker = YOLOAttributeTrackerOptimized(
        model_path=args.model_path,
        conf_threshold=args.conf_threshold,
        attribute_model_path=args.attribute_model_path,
        use_gpu=use_gpu,
        use_multithreading=use_multithreading,
        num_workers=num_workers
    )
    tracker.process_video(args.video_path, output_path=args.output_path, display=args.display, save_data=save_data)
    
    if args.output_path:
        print(f"\nOutput video saved to: {args.output_path}")

if __name__ == "__main__":
    main()
