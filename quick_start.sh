#!/bin/bash

# People Attribute Tracker Quick Start Script for Linux/macOS

echo "========================================"
echo "People Attribute Tracker - Quick Start"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created successfully."
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Check if models exist
if [ ! -f "yolov8n.pt" ]; then
    echo ""
    echo "Warning: YOLOv8 model not found."
    echo "The model will be downloaded automatically on first run."
    echo ""
fi

# Run example
echo ""
echo "========================================"
echo "Running example with test video..."
echo "========================================"
echo ""

python tracker.py examples/测试视频.mp4 output/example_output.mp4

echo ""
echo "========================================"
echo "Processing completed!"
echo "========================================"
echo ""
echo "Output files:"
echo "- Video: output/example_output.mp4"
echo "- Data: output/*.csv, output/*.json"
echo ""
echo "To visualize results, run:"
echo "python visualize.py --data output/*.csv --all"
echo ""
