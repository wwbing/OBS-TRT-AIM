# YOLO TensorRT OBS Filter — 架构文档

## 概述

OBS Studio 视频滤镜插件，在显示器采集画面中心 640×640 区域实时运行 YOLO TensorRT FP16 推理，检测敌人并绘制绿色边界框。

- **模型**: YOLO26 end-to-end，输入 `1×3×640×640 FP32`，输出 `1×300×6 FP32`（无需 NMS）
- **推理引擎**: TensorRT 10.16 + CUDA 13.1，FP16 内部计算
- **帧率**: 120fps OBS 渲染管线
- **硬件**: RTX 5070 Ti 16GB

---

## 整体管线

```
OBS 渲染线程 (filter_video_render)
│
├─ [GPU] 1. 渲染原始画面
│         obs_source_process_filter_begin/end
│
├─ [GPU] 2. 中心裁剪渲染到 640×640 texrender
│         gs_texrender_begin → gs_ortho(cx, cx+640, cy, cy+640) → obs_source_video_render
│         数据量: 640×640×4 = 1.6MB BGRA8
│
├─ [GPU] 3. D3D11 CopyResource
│         将 texrender 的 D3D11 纹理拷贝到自建的 shared 纹理（带 MISC_SHARED 标志）
│         数据量: 1.6MB GPU 内部拷贝
│
├─ [GPU] 4. CUDA-D3D11 Interop
│         cudaGraphicsMapResources → cudaGraphicsSubResourceGetMappedArray
│         → cudaMemcpy2DFromArrayAsync(DeviceToDevice) → d_bgra
│         → cudaGraphicsUnmapResources
│         数据量: 1.6MB GPU 内部拷贝
│
├─ [CPU] 5. cudaDeviceSynchronize
│         （driver API cuLaunchKernel 需要与 runtime API 同步）
│
├─ [GPU] 6. NVRTC Kernel
│         cuLaunchKernel "bgra_to_fp32_chw": BGRA8 → FP32 CHW 归一化
│         数据量: 1.6MB → 4.9MB
│
├─ [CPU] 7. cudaDeviceSynchronize
│         （driver API kernel 需要 barrier 才进入 TRT）
│
├─ [GPU] 8. TensorRT 推理
│         enqueueV3 (FP16 内部计算)
│         数据量: 4.9MB 输入, 7.2KB 输出
│         实测: ~1.5ms
│
├─ [GPU→CPU] 9. 回读检测结果
│            cudaMemcpy DeviceToHost: 7.2KB
│
├─ [CPU] 10. Postprocess
│         坐标映射 (模型 640 坐标 → 全帧坐标)
│
└─ [GPU] 11. 画框
          gs_render_start/stop 绘制绿色矩形框
```

---

## 各步骤详述

### 1. 原始画面渲染

```cpp
obs_source_process_filter_begin(f->context, GS_BGRA, OBS_ALLOW_DIRECT_RENDERING);
obs_source_process_filter_end(f->context, obs_get_base_effect(OBS_EFFECT_DEFAULT), 0, 0);
```

OBS 标准滤镜前置渲染。渲染父源到当前渲染目标。

### 2. 中心裁剪渲染

```cpp
gs_texrender_begin(f->texrender, 640, 640);
gs_ortho((float)cx, (float)(cx + 640), (float)cy, (float)(cy + 640), -1.0f, 1.0f);
obs_source_video_render(parent);
gs_texrender_end(f->texrender);
```

**关键设计**：`gs_ortho` 映射的是父源坐标 `(cx, cx+640)` → texrender 的 `(0, 640)`，实现 1:1 像素映射的中心裁剪。**不会拉伸或变形**。

### 3-4. D3D11→CUDA 传输

```
texrender D3D11纹理 → CopyResource → 自建 shared 纹理 → CUDA map → GPU memcpy → d_bgra
```

- texrender 的纹理由 OBS 创建，**没有** `D3D11_RESOURCE_MISC_SHARED`，CUDA 无法直接注册
- 自建 640×640 `ID3D11Texture2D` 带 `MISC_SHARED` 标志，注册给 CUDA
- `CopyResource` 是 D3D11 标准 GPU 拷贝 API
- `cudaMemcpy2DFromArrayAsync(DeviceToDevice)` 是 GPU 内部的 array → linear 拷贝
- **全程不经过 CPU**

### 5-7. NVRTC 动态编译 Kernel

```cuda
__global__ void bgra_to_fp32_chw(
    const unsigned char *bgra, int w, int h,
    float *output, int cropX, int cropY)
{
    // BGRA → FP32 CHW, 归一化到 [0, 1], BGR→RGB 通道交换
    output[0 * 640*640 + idx] = (float)bgra[R] / 255.0f;
    output[1 * 640*640 + idx] = (float)bgra[G] / 255.0f;
    output[2 * 640*640 + idx] = (float)bgra[B] / 255.0f;
}
```

运行时通过 NVRTC 编译，无需 `nvcc` 或 `.cu` 文件。编译选项: `-arch=compute_89`。

### 8. TensorRT 推理

```cpp
f->execCtx->setInputTensorAddress(name, d_input);
f->execCtx->setOutputTensorAddress(name, d_output);
f->execCtx->enqueueV3(nullptr);  // nullptr = default CUDA stream
```

- 使用 TRT 10.x IOTensor API，动态查询 bindings 维度和类型
- 输入: `1×3×640×640 FP32` (4.9MB)
- 输出: `1×300×6 FP32` (7.2KB) — `[x1, y1, x2, y2, conf, cls]`

### 9-10. 后处理

```cpp
float s = (float)cropW / 640.0f;  // = 1.0 for 640×640 crop
x_full = x_model * s + cropX;
y_full = y_model * s + cropY;
```

模型输出在 640×640 坐标空间。乘以 `cropW/640` 映射到裁剪区，加 `cropX` 偏移到全帧坐标。

### 11. 画框

使用 OBS graphics API (`gs_render_start` / `gs_vertex2f` / `gs_render_stop` + `GS_TRISTRIP`) 绘制绿色矩形框。在 `gs_ortho(0, w, 0, h)` 像素坐标系下绘制。

---

## 关键设计决策

### 为什么在 video_render 而不是 video_tick 做推理？

`video_tick` 在同步源的 OBS 渲染线程中**不能调用图形 API**，否则导致设备丢失或黑屏。所有渲染和 CUDA 操作必须在 `video_render` 中完成。

### 为什么用默认 CUDA stream (nullptr)？

`cudaStream_t` 在 `filter_create`（OBS 初始化线程）创建后，无法在 `filter_video_render`（OBS 渲染线程）安全使用。跨线程共享 stream 导致 "isStreamCapturing" 错误。默认流由 CUDA 运行时自动管理线程绑定。

### 为什么用 driver API 的 cuLaunchKernel？

TRT 10.x 的 NVRTC 编译产物是 PTX，需要用 `cuModuleLoadData` + `cuModuleGetFunction` + `cuLaunchKernel` 加载。这导致了与 CUDA runtime API 的混用，需要 `cudaDeviceSynchronize` 做 barrier。

---

## 性能瓶颈

| 瓶颈 | 位置 | 影响 |
|------|------|------|
| **cudaDeviceSynchronize ×2** | 步骤 5 + 7 | 每帧空等 GPU 完成，阻塞渲染线程 |
| **全流程在渲染线程** | 整体架构 | ~2-3ms 阻塞 / 帧, 占 120fps 预算 25-36% |
| **CopyResource + CUDA memcpy** | 步骤 3 + 5 | 虽然 GPU 内部，但增加了一步多余的显存搬运 |

实测 GPU 推理耗时 **~1.5ms**（OBS 日志），端到端渲染线程阻塞约 **2-3ms**。

---

## 未来优化方向

### 1. 消除两次 cudaDeviceSynchronize（短期，优先）

**方案 A**: 用 CUDA runtime API (`cudaLaunchKernel` 或 `cudaLaunchCooperativeKernel`) 替代 `cuLaunchKernel`，全程 runtime API，无需 barrier。

**方案 B**: 把 NVRTC kernel 换成 CUDA Runtime Compilation + `cuModuleLoadDataEx`，加载到 runtime context。

**预期收益**: 减少 0.05-0.1ms 阻塞。

### 2. 异步推理流水线（中期）

将 TRT 推理移到独立 CUDA stream，与渲染线程解耦：

```
帧 N:   渲染 → 截帧 → 提交推理(stream) → 用帧 N-1 的结果画框 → 返回
帧 N+1: 渲染 → 截帧 → 提交推理(stream) → 用帧 N 的结果画框   → 返回
```

渲染线程不再等待 TRT 完成，仅提交任务后立即返回。牺牲 1 帧延迟换取零阻塞。

**预期收益**: 渲染线程阻塞降为 < 0.5ms。

### 3. 创建 CUDA 兼容的 gs_texture 作为渲染目标（长期）

自建 `gs_texture_t`，底层 D3D11 纹理带 `MISC_SHARED` + `BIND_RENDER_TARGET`。用 OBS 的 `gs_set_render_target` 直接渲染到该纹理，省掉 texrender → CopyResource 这一步。

**预期收益**: 减少 1.6MB 的 GPU 拷贝。

### 4. 使用 NVIDIA NPP 库替代 NVRTC kernel

NPP (NVIDIA Performance Primitives) 提供预编译的 `nppiResize` / `nppiColorTwist` 等函数，替代手写 kernel。无需运行时编译，直接用 runtime API 调用，消除 driver/runtime API 混用问题。
