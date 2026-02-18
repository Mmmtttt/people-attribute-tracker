@echo off
REM People Attribute Tracker Quick Start Script for Windows

echo ========================================
echo People Attribute Tracker - Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created successfully.
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

REM Check if models exist
if not exist "yolov8n.pt" (
    echo.
    echo Warning: YOLOv8 model not found.
    echo The model will be downloaded automatically on first run.
    echo.
)

REM Run example
echo.
echo ========================================
echo Running example with test video...
echo ========================================
echo.

python tracker.py examples/测试视频.mp4 output/example_output.mp4

echo.
echo ========================================
echo Processing completed!
echo ========================================
echo.
echo Output files:
echo - Video: output/example_output.mp4
echo - Data: output/*.csv, output/*.json
echo.
echo To visualize results, run:
echo python visualize.py --data output/*.csv --all
echo.

pause
