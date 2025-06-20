# FreeTimeGS 4D Gaussian Splatting Viewer

A streamlined implementation of FreeTimeGS for 4D Gaussian Splatting with web-based visualization.

## Quick Start

1. **Add your video** to the `videos/` directory
2. **Run the pipeline**:
   ```bash
   python run_pipeline.py
   ```
3. **Open** http://localhost:8000 in your browser

## Options

- **Demo mode** (no video needed): `python run_pipeline.py --skip-training`
- **Custom video**: `python run_pipeline.py --video path/to/video.mp4`
- **Fast training**: `python run_pipeline.py --iterations 1000`

## Pipeline

1. **Extract frames** from video at 10 FPS (~300 frames)
2. **Estimate camera poses** using COLMAP (optional)
3. **Train 4D Gaussians** with temporal dynamics
4. **Launch web viewer** with interactive controls

## Web Viewer Features

- **Temporal scrubbing**: Drag time slider to see 4D evolution
- **Play/pause**: Automatic temporal animation
- **Mouse controls**: Rotate camera, zoom with scroll
- **Real-time**: Smooth 60fps WebGL rendering

## Requirements

- Python 3.12+
- PyTorch (CPU/GPU)
- OpenCV
- FastAPI
- COLMAP (optional, for real camera poses)

Built with `uv` for fast dependency management.
