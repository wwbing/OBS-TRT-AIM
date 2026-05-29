# YOLO TensorRT Filter for OBS Studio

实时目标检测 OBS 滤镜插件，基于 TensorRT + CUDA 在显示器采集画面上运行 YOLO 推理，绘制检测框。

## 运行环境

| 组件 | 版本 |
|------|------|
| OBS Studio | 32.1.1 (Steam) |
| CUDA Toolkit | 13.1 |
| TensorRT | 10.16 |
| cuDNN | 9.22 |
| Visual Studio | 2026 Community |
| CMake | 4.3+ |
| GPU | RTX 5070 Ti（或同等级） |

## 快速开始

### 1. 下载依赖

首次构建需要手动下载 3 个 zip 包放到 `.deps/` 目录（GitHub 直连可能较慢）：

| 文件 | URL |
|------|-----|
| `windows-deps-2025-07-11-x64.zip` | `https://github.com/obsproject/obs-deps/releases/download/2025-07-11/` |
| `windows-deps-qt6-2025-07-11-x64.zip` | `https://github.com/obsproject/obs-deps/releases/download/2025-07-11/` |
| `31.1.1.zip` | `https://github.com/obsproject/obs-studio/archive/refs/tags/` |

```bash
mkdir .deps
# 将下载好的 3 个文件放入 .deps/
```

### 2. 构建插件

```powershell
# 配置 (初次约 15 分钟，后续不到 1 分钟)
cmake --preset windows-x64

# 编译
cmake --build build_x64 --config RelWithDebInfo
```

### 3. 安装到 OBS

关闭 OBS，然后：

```powershell
cp build_x64/RelWithDebInfo/plugintemplate-for-obs.dll "D:/steam/steamapps/common/OBS Studio/obs-plugins/64bit/"
cp -r build_x64/rundir/RelWithDebInfo/plugintemplate-for-obs "D:/steam/steamapps/common/OBS Studio/obs-plugins/64bit/"
```

### 4. 使用

1. 打开 OBS，在「显示器采集」源上右键 → **滤镜**
2. 点击 + → **YOLO TensorRT Filter**
3. Engine Path 默认已指向 `D:/code/obs-plugintemplate/data/best_trt.engine`
4. 调整 Confidence Threshold（推荐 0.25）
5. 确认后预览画面中心区域出现检测框

## 架构

详见 [doc/architecture.md](doc/architecture.md)

- **OBS 滤镜类型**: `OBS_SOURCE_TYPE_FILTER`，运行在 `filter_video_render`
- **管线**: 画面渲染 → 640×640 中心裁剪 → D3D11→CUDA 互操作 → NVRTC Kernel → TensorRT 推理 → 画框
- **全程 GPU**: 除 7.2KB 检测结果回读外，所有图像数据不经过 CPU

## 文件结构

```
├── src/
│   ├── plugin-main.cpp      # 主插件代码
│   ├── debug_utils.h/.cpp   # 调试帧保存（可禁用）
│   ├── stb_image_resize2.h  # STB 图像缩放（未使用，保留备用）
│   └── stb_image_write.h    # 调试帧 JPEG 输出
├── data/
│   └── best_trt.engine      # TensorRT 引擎文件
├── CMakeLists.txt
├── CMakePresets.json
├── buildspec.json
└── doc/
    └── architecture.md      # 详细架构文档
```

## 调试

如需查看送入模型的截图，打开 `src/debug_utils.h`，将 `kDebugDumpFrames` 改为 `true`。每 120 帧保存一张 640×640 JPEG 到 `data/` 目录。
