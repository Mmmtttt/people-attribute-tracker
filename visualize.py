import argparse
import os
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from collections import defaultdict
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AttributeVisualizer:
    def __init__(self, csv_path, json_path=None, output_dir='output'):
        self.csv_path = csv_path
        self.json_path = json_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.data = self.load_data()
        self.summary = self.load_summary()
        
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def load_data(self):
        df = pd.read_csv(self.csv_path)
        return df
    
    def load_summary(self):
        if self.json_path and os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def generate_all_charts(self, chart_types=None):
        if chart_types is None:
            chart_types = ['gender', 'age', 'orientation', 'accessory', 
                          'bag', 'upper', 'lower', 'overview']
        
        print(f"Generating charts: {', '.join(chart_types)}")
        
        for chart_type in chart_types:
            try:
                if chart_type == 'gender':
                    self.plot_gender_distribution()
                elif chart_type == 'age':
                    self.plot_age_distribution()
                elif chart_type == 'orientation':
                    self.plot_orientation_distribution()
                elif chart_type == 'accessory':
                    self.plot_accessory_distribution()
                elif chart_type == 'bag':
                    self.plot_bag_distribution()
                elif chart_type == 'upper':
                    self.plot_upper_style_distribution()
                elif chart_type == 'lower':
                    self.plot_lower_style_distribution()
                elif chart_type == 'overview':
                    self.plot_overview()
            except Exception as e:
                print(f"Error generating {chart_type} chart: {e}")
        
        print(f"All charts saved to {self.output_dir}/")
    
    def plot_gender_distribution(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        gender_counts = self.data['gender'].value_counts()
        
        colors = ['#FF6B6B', '#4ECDC4']
        ax1.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90, explode=(0.05, 0))
        ax1.set_title('性别分布', fontsize=14, fontweight='bold')
        
        bars = ax2.bar(gender_counts.index, gender_counts.values, color=colors)
        ax2.set_xlabel('性别', fontsize=12)
        ax2.set_ylabel('人数', fontsize=12)
        ax2.set_title('性别统计', fontsize=14, fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'gender_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_age_distribution(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        age_counts = self.data['age'].value_counts()
        age_order = ['小于18岁', '18-60岁', '大于60岁']
        age_counts = age_counts.reindex(age_order, fill_value=0)
        
        colors = ['#95E1D3', '#F38181', '#AA96DA']
        bars = ax1.bar(age_counts.index, age_counts.values, color=colors)
        ax1.set_xlabel('年龄段', fontsize=12)
        ax1.set_ylabel('人数', fontsize=12)
        ax1.set_title('年龄分布', fontsize=14, fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=11)
        
        ax2.pie(age_counts.values, labels=age_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax2.set_title('年龄比例', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'age_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_orientation_distribution(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        orientation_counts = self.data['orientation'].value_counts()
        colors = ['#FFEAA7', '#DDA0DD', '#98D8C8']
        bars = ax.bar(orientation_counts.index, orientation_counts.values, color=colors)
        
        ax.set_xlabel('朝向', fontsize=12)
        ax.set_ylabel('人数', fontsize=12)
        ax.set_title('人员朝向分布', fontsize=14, fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'orientation_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_accessory_distribution(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        accessory_counts = self.data['accessory'].value_counts()
        colors = ['#F8B500', '#00CED1', '#FF69B4']
        bars = ax.bar(accessory_counts.index, accessory_counts.values, color=colors)
        
        ax.set_xlabel('配饰', fontsize=12)
        ax.set_ylabel('人数', fontsize=12)
        ax.set_title('配饰分布', fontsize=14, fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'accessory_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_bag_distribution(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bag_counts = self.data['bag'].value_counts()
        colors = ['#FF6F61', '#6B5B95', '#88B04B']
        bars = ax.bar(bag_counts.index, bag_counts.values, color=colors)
        
        ax.set_xlabel('包类型', fontsize=12)
        ax.set_ylabel('人数', fontsize=12)
        ax.set_title('包类型分布', fontsize=14, fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'bag_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_upper_style_distribution(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        upper_counts = self.data['upper_style'].value_counts()
        colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD']
        bars = ax.bar(upper_counts.index, upper_counts.values, color=colors)
        
        ax.set_xlabel('上衣风格', fontsize=12)
        ax.set_ylabel('人数', fontsize=12)
        ax.set_title('上衣风格分布', fontsize=14, fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'upper_style_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_lower_style_distribution(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        lower_counts = self.data['lower_style'].value_counts()
        colors = ['#20B2AA', '#FF7F50']
        bars = ax.bar(lower_counts.index, lower_counts.values, color=colors)
        
        ax.set_xlabel('下装风格', fontsize=12)
        ax.set_ylabel('人数', fontsize=12)
        ax.set_title('下装风格分布', fontsize=14, fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'lower_style_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_overview(self):
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        gender_counts = self.data['gender'].value_counts()
        ax1.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
        ax1.set_title('性别分布')
        
        ax2 = fig.add_subplot(gs[0, 1])
        age_counts = self.data['age'].value_counts()
        age_order = ['小于18岁', '18-60岁', '大于60岁']
        age_counts = age_counts.reindex(age_order, fill_value=0)
        ax2.bar(age_counts.index, age_counts.values, color=['#95E1D3', '#F38181', '#AA96DA'])
        ax2.set_title('年龄分布')
        ax2.set_ylabel('人数')
        
        ax3 = fig.add_subplot(gs[0, 2])
        orientation_counts = self.data['orientation'].value_counts()
        ax3.bar(orientation_counts.index, orientation_counts.values, color=['#FFEAA7', '#DDA0DD', '#98D8C8'])
        ax3.set_title('朝向分布')
        ax3.set_ylabel('人数')
        
        ax4 = fig.add_subplot(gs[1, 0])
        accessory_counts = self.data['accessory'].value_counts()
        ax4.bar(accessory_counts.index, accessory_counts.values, color=['#F8B500', '#00CED1', '#FF69B4'])
        ax4.set_title('配饰分布')
        ax4.set_ylabel('人数')
        
        ax5 = fig.add_subplot(gs[1, 1])
        bag_counts = self.data['bag'].value_counts()
        ax5.bar(bag_counts.index, bag_counts.values, color=['#FF6F61', '#6B5B95', '#88B04B'])
        ax5.set_title('包类型分布')
        ax5.set_ylabel('人数')
        
        ax6 = fig.add_subplot(gs[1, 2])
        upper_counts = self.data['upper_style'].value_counts()
        ax6.bar(upper_counts.index, upper_counts.values, color=['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD'])
        ax6.set_title('上衣风格分布')
        ax6.set_ylabel('人数')
        
        ax7 = fig.add_subplot(gs[2, :])
        lower_counts = self.data['lower_style'].value_counts()
        ax7.bar(lower_counts.index, lower_counts.values, color=['#20B2AA', '#FF7F50'])
        ax7.set_title('下装风格分布')
        ax7.set_ylabel('人数')
        
        if self.summary:
            fig.suptitle(f"人员属性识别统计概览\n总人数: {len(self.data)}", 
                        fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'overview.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate visualization charts for attribute tracking results')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--json', type=str, help='Path to JSON summary file')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for charts')
    parser.add_argument('--charts', type=str, nargs='+', 
                       choices=['gender', 'age', 'orientation', 'accessory', 'bag', 'upper', 'lower', 'overview'],
                       help='Chart types to generate')
    parser.add_argument('--all', action='store_true', help='Generate all charts')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        return
    
    json_path = args.json
    if not json_path:
        json_dir = os.path.dirname(args.data) if os.path.dirname(args.data) else '.'
        json_files = [f for f in os.listdir(json_dir) if f.startswith('yolo_attribute_summary_') and f.endswith('.json')]
        if json_files:
            json_path = os.path.join(json_dir, json_files[0])
            print(f"Auto-detected JSON file: {json_path}")
    
    visualizer = AttributeVisualizer(args.data, json_path, args.output_dir)
    
    if args.all or not args.charts:
        visualizer.generate_all_charts()
    else:
        visualizer.generate_all_charts(args.charts)

if __name__ == '__main__':
    main()
