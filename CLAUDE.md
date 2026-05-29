# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```powershell
# Configure (first time or after CMakeLists.txt changes)
cd D:/code/obs-plugintemplate; ./rm -r -fo build_x64; cmake --preset windows-x64

# Build
cmake --build build_x64 --config RelWithDebInfo

# Install to Steam OBS (close OBS first)
cp build_x64/RelWithDebInfo/plugintemplate-for-obs.dll "D:/steam/steamapps/common/OBS Studio/obs-plugins/64bit/"
cp -r build_x64/rundir/RelWithDebInfo/plugintemplate-for-obs "D:/steam/steamapps/common/OBS Studio/obs-plugins/64bit/"
```

## Architecture

This is an **OBS Studio filter plugin** that runs YOLO TensorRT FP16 inference on video frames. It detects objects on the user's display capture and draws green bounding boxes.

**Pipeline** (all in `filter_video_render`):
1. `obs_source_process_filter_begin/end` — render original source
2. `gs_texrender` + `gs_stagesurface_map` — GPU→CPU readback of 640×640 center crop (1.6MB)
3. CPU BGRA→FP16 conversion with IEEE 754 float32→float16
4. `cudaMemcpy` to GPU → `executeV2` TRT inference
5. `cudaMemcpy` detections back (7.2KB) → postprocess
6. `gs_render_start/stop` — draw green bounding boxes via OBS graphics API

**Key constraint**: `video_tick` must NOT call any graphics/CUDA APIs. All rendering and inference work belongs in `video_render`. CUDA operations run on the default stream (NULL) because the OBS graphics thread cannot share a `cudaStream_t` created on the init thread.

**Model**: YOLO26 end-to-end (no NMS needed). Input: 640×640×3 FP16 CHW. Output: 300×6 FP32 `[x1,y1,x2,y2,conf,cls]`.

**Engine file** at `data/best_trt.engine` — copied to OBS plugin dir on install.

**Environment paths** (per `train_custom_dataset.md`):
- CUDA 13.1: `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1`
- TensorRT 10.16: `C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.16.1.11`
- VS 2026 (v18): `C:/Program Files/Microsoft Visual Studio/18`
- Windows SDK 10.0.26100
- OBS Studio (Steam): `D:/steam/steamapps/common/OBS Studio`

**Dependencies**: CMake buildspec auto-downloads obs-deps + obs-studio source into `.deps/` and builds libobs. On first configure, download three zip files manually if GitHub is slow: `windows-deps-2025-07-11-x64.zip`, `windows-deps-qt6-2025-07-11-x64.zip`, `31.1.1.zip` → place in `.deps/`.
