#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <nvrtc.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <windows.h>
#include <d3d11.h>
#include <obs-module.h>
#include <plugin-support.h>

#include "debug_utils.h"
#include "mouse_controller.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

using namespace nvinfer1;

static constexpr int kInputW = 640;
static constexpr int kInputH = 640;

struct Detection { float x1, y1, x2, y2, conf; int cls; };

struct TRTLogger : ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) blog(LOG_WARNING, "[TRT] %s", msg);
    }
};
static TRTLogger gLogger;

// NVRTC kernel: GPU BGRA8 → FP32 CHW
static const char *kKernelSrc = R"(
extern "C" __global__ void bgra_to_fp32_chw(
    const unsigned char *__restrict__ bgra, int texW, int texH,
    float *__restrict__ output, int cropX, int cropY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= 640 || y >= 640) return;
    int srcX = cropX + x;
    int srcY = cropY + y;
    int srcIdx = (srcY * texW + srcX) * 4;
    int dstIdx = y * 640 + x;
    int base = 640 * 640;
    output[0 * base + dstIdx] = (float)bgra[srcIdx + 2] / 255.0f;
    output[1 * base + dstIdx] = (float)bgra[srcIdx + 1] / 255.0f;
    output[2 * base + dstIdx] = (float)bgra[srcIdx + 0] / 255.0f;
}
)";

struct filter_data {
    obs_source_t *context;

    // TensorRT
    IRuntime *runtime = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *execCtx = nullptr;

    std::vector<void *> trtBindings;
    void *d_input = nullptr;
    void *d_output = nullptr;
    size_t outputBytes = 0;

    void *d_bgra = nullptr;  // 640*640*4 GPU buffer

    // GPU texrender (640x640 center crop)
    gs_texrender_t *texrender = nullptr;

    // CUDA-D3D11 interop: our own shared texture, CopyResource from texrender
    ID3D11Texture2D *sharedTex = nullptr;
    cudaGraphicsResource_t cudaRes = nullptr;

    // NVRTC
    CUmodule cuModule = nullptr;
    CUfunction cuKernel = nullptr;

    float confThreshold = 0.25f;
    bool engineLoaded = false;

    // Mouse aim assist
    MouseController mouse;
    bool aimAssistEnabled = false;
    float aimSensitivity = 1.0f;

    cudaEvent_t evStart = nullptr, evStop = nullptr;
    float gpuLatencyMs = 0.0f;

    std::vector<Detection> detections;
    int tickCount = 0, inferCount = 0;
};

// ----------------------------------------------------------------
static bool compileKernel(filter_data *f) {
    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, kKernelSrc, "kern.cu", 0, nullptr, nullptr)
        != NVRTC_SUCCESS) {
        blog(LOG_ERROR, "[YOLO] nvrtcCreateProgram failed"); return false;
    }
    const char *opts[] = {"-arch=compute_89", "-std=c++17",
        "-IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include"};
    if (nvrtcCompileProgram(prog, 3, opts) != NVRTC_SUCCESS) {
        size_t sz; nvrtcGetProgramLogSize(prog, &sz);
        std::string log(sz, '\0'); nvrtcGetProgramLog(prog, log.data());
        blog(LOG_ERROR, "[YOLO] NVRTC error:\n%s", log.c_str());
        nvrtcDestroyProgram(&prog); return false;
    }
    size_t ptxSize; nvrtcGetPTXSize(prog, &ptxSize);
    std::vector<char> ptx(ptxSize);
    nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);

    if (cuModuleLoadData(&f->cuModule, ptx.data()) != CUDA_SUCCESS
        || cuModuleGetFunction(&f->cuKernel, f->cuModule,
                               "bgra_to_fp32_chw") != CUDA_SUCCESS) {
        blog(LOG_ERROR, "[YOLO] cuModuleLoad/GetFunction failed"); return false;
    }
    blog(LOG_INFO, "[YOLO] NVRTC kernel OK (FP32)");
    return true;
}

// ----------------------------------------------------------------
static std::vector<char> loadEngineFile(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { blog(LOG_ERROR, "[YOLO] open: %s", path.c_str()); return {}; }
    size_t size = f.tellg(); f.seekg(0);
    std::vector<char> data(size);
    f.read(data.data(), size);
    blog(LOG_INFO, "[YOLO] Engine: %s (%.1f MB)", path.c_str(), size / 1048576.0);
    return data;
}

static bool initTRT(filter_data *f, const char *enginePath) {
    auto data = loadEngineFile(enginePath);
    if (data.empty()) return false;
    f->runtime = createInferRuntime(gLogger);
    f->engine = f->runtime->deserializeCudaEngine(data.data(), data.size());
    f->execCtx = f->engine->createExecutionContext();

    int nb = f->engine->getNbIOTensors();
    f->trtBindings.resize(nb, nullptr);

    for (int i = 0; i < nb; i++) {
        const char *name = f->engine->getIOTensorName(i);
        Dims dims = f->engine->getTensorShape(name);
        DataType dt = f->engine->getTensorDataType(name);
        TensorIOMode mode = f->engine->getTensorIOMode(name);

        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; d++)
            vol *= (dims.d[d] > 0 ? dims.d[d] : 1);
        size_t bytes = vol * ((dt == DataType::kFLOAT) ? 4 : 2);

        blog(LOG_INFO, "[YOLO] IO[%d] '%s': dims=%s type=%s mode=%s vol=%zu bytes=%zu",
             i, name,
             [&]() { std::string s; for(int d=0;d<dims.nbDims;d++){if(d)s+="x";s+=std::to_string(dims.d[d]);} return s; }().c_str(),
             (dt == DataType::kFLOAT) ? "FP32" : "FP16",
             (mode == TensorIOMode::kINPUT) ? "IN" : "OUT",
             vol, bytes);

        cudaMalloc(&f->trtBindings[i], bytes);

        if (mode == TensorIOMode::kINPUT) {
            f->d_input = f->trtBindings[i];
            f->execCtx->setInputTensorAddress(name, f->trtBindings[i]);
        } else {
            f->d_output = f->trtBindings[i];
            f->outputBytes = bytes;
            f->execCtx->setOutputTensorAddress(name, f->trtBindings[i]);
        }
    }

    cudaMalloc(&f->d_bgra, kInputW * kInputH * 4);
    cudaEventCreate(&f->evStart);
    cudaEventCreate(&f->evStop);

    obs_enter_graphics();
    f->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
    obs_leave_graphics();

    if (!compileKernel(f)) return false;
    blog(LOG_INFO, "[YOLO] TRT + NVRTC ready (FP32, dynamic bindings)");
    return true;
}

// ----------------------------------------------------------------
static const char *filter_get_name(void *) { return "YOLO TensorRT Filter"; }

static void *filter_create(obs_data_t *settings, obs_source_t *context) {
    auto *f = new filter_data();
    f->context = context;
    const char *path = obs_data_get_string(settings, "engine_path");
    f->confThreshold = (float)obs_data_get_double(settings, "conf_threshold");
    blog(LOG_INFO, "[YOLO] create: engine=%s conf=%.2f",
         path ? path : "(null)", f->confThreshold);
    if (path && *path && initTRT(f, path)) f->engineLoaded = true;
    return f;
}

static void filter_destroy(void *data) {
    auto *f = (filter_data *)data;
    if (f->cudaRes) cudaGraphicsUnregisterResource(f->cudaRes);
    if (f->sharedTex) f->sharedTex->Release();
    if (f->cuModule) cuModuleUnload(f->cuModule);
    obs_enter_graphics();
    if (f->texrender) gs_texrender_destroy(f->texrender);
    obs_leave_graphics();
    cudaEventDestroy(f->evStart); cudaEventDestroy(f->evStop);
    for (void *ptr : f->trtBindings) { if (ptr) cudaFree(ptr); }
    cudaFree(f->d_bgra);
    delete f->execCtx; delete f->engine; delete f->runtime;
    blog(LOG_INFO, "[YOLO] destroy (infer=%d)", f->inferCount);
    delete f;
}

static void filter_update(void *data, obs_data_t *settings) {
    auto *f = (filter_data *)data;
    f->confThreshold = (float)obs_data_get_double(settings, "conf_threshold");
    f->aimAssistEnabled = obs_data_get_bool(settings, "aim_assist");
    f->aimSensitivity = (float)obs_data_get_double(settings, "aim_sensitivity");

    // Lazy-load mouse DLL when aim assist is first enabled
    if (f->aimAssistEnabled && !f->mouse.IsLoaded()) {
        const char *paths[] = {
            "D:/steam/steamapps/common/OBS Studio/obs-plugins/64bit/plugintemplate-for-obs/ddll64.dll",
            "D:/code/obs-plugintemplate/data/ddll64.dll",
        };
        char *modPath = obs_module_file("ddll64.dll");
        if (modPath) { f->mouse.Load(modPath); bfree(modPath); }
        if (!f->mouse.IsLoaded()) {
            for (auto p : paths) {
                if (f->mouse.Load(p)) { blog(LOG_INFO, "[YOLO] Mouse DLL: %s", p); break; }
            }
        }
        if (!f->mouse.IsLoaded()) blog(LOG_WARNING, "[YOLO] Mouse DLL load FAILED");
    }
}

// ----------------------------------------------------------------
static std::vector<Detection> postprocess(const float *output, int texW, int texH,
                                           int cropW, int cropH,
                                           int cropX, int cropY, float confThr) {
    std::vector<Detection> dets;
    // Scale from model 640 coords → crop region coords, then offset to full frame
    float s = (float)cropW / (float)kInputW;
    for (int i = 0; i < 300; i++) {
        float x1 = output[i*6+0], y1 = output[i*6+1];
        float x2 = output[i*6+2], y2 = output[i*6+3];
        float conf = output[i*6+4]; int cls = (int)output[i*6+5];
        if (conf < confThr) continue;
        x1 = std::max(0.0f, std::min(x1 * s + cropX, (float)texW - 1));
        y1 = std::max(0.0f, std::min(y1 * s + cropY, (float)texH - 1));
        x2 = std::max(0.0f, std::min(x2 * s + cropX, (float)texW - 1));
        y2 = std::max(0.0f, std::min(y2 * s + cropY, (float)texH - 1));
        dets.push_back({x1, y1, x2, y2, conf, cls});
    }
    return dets;
}

// ----------------------------------------------------------------
static void filter_video_tick(void *data, float seconds) {
    auto *f = (filter_data *)data;
    f->tickCount++;
}

// ----------------------------------------------------------------
static void filter_video_render(void *data, gs_effect_t *effect) {
    auto *f = (filter_data *)data;

    // Step 1: Render original source
    obs_source_process_filter_begin(f->context, GS_BGRA, OBS_ALLOW_DIRECT_RENDERING);
    obs_source_process_filter_end(f->context, obs_get_base_effect(OBS_EFFECT_DEFAULT), 0, 0);

    if (!f->engineLoaded) return;
    obs_source_t *parent = obs_filter_get_parent(f->context);
    if (!parent) return;
    uint32_t w = obs_source_get_width(parent);
    uint32_t h = obs_source_get_height(parent);
    if (w == 0 || h == 0) return;

    int cw = std::min(kInputW, (int)w), ch = std::min(kInputH, (int)h);
    int cx = ((int)w - cw) / 2, cy = ((int)h - ch) / 2;

    // Step 2: Render parent to 640x640 texrender with center crop ortho
    // This PREVENTS stretching — only the center 640x640 region maps to the texrender
    gs_texrender_reset(f->texrender);
    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
    struct vec4 cl; vec4_zero(&cl);

    if (gs_texrender_begin(f->texrender, kInputW, kInputH)) {
        gs_clear(GS_CLEAR_COLOR, &cl, 0.0f, 0);
        // Center crop: maps parent coords (cx..cx+640, cy..cy+640) → texrender (0..640, 0..640)
        gs_ortho((float)cx, (float)(cx + kInputW),
                 (float)cy, (float)(cy + kInputH), -1.0f, 1.0f);
        gs_matrix_identity();
        obs_source_video_render(parent);
        gs_texrender_end(f->texrender);
    }
    gs_blend_state_pop();

    // Step 3: Get texrender's D3D11 texture, create our shared tex, CopyResource
    gs_texture_t *gsTex = gs_texrender_get_texture(f->texrender);
    if (!gsTex) return;
    ID3D11Texture2D *srcTex = (ID3D11Texture2D *)gs_texture_get_obj(gsTex);
    if (!srcTex) return;

    // Lazy-create shared texture
    if (!f->sharedTex) {
        ID3D11Device *d3dDev = (ID3D11Device *)gs_get_device_obj();
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = kInputW; desc.Height = kInputH;
        desc.MipLevels = 1; desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

        if (FAILED(d3dDev->CreateTexture2D(&desc, nullptr, &f->sharedTex))) {
            blog(LOG_WARNING, "[YOLO] CreateTexture2D failed"); return;
        }
        cudaError_t err = cudaGraphicsD3D11RegisterResource(&f->cudaRes, f->sharedTex,
                                                             cudaGraphicsRegisterFlagsNone);
        if (err != cudaSuccess) {
            blog(LOG_WARNING, "[YOLO] CUDA interop reg: %s", cudaGetErrorString(err));
            f->sharedTex->Release(); f->sharedTex = nullptr; return;
        }
        blog(LOG_INFO, "[YOLO] CUDA-D3D11 interop OK");
    }

    // Copy texrender → our shared texture (D3D11 GPU copy)
    ID3D11Device *d3dDev = (ID3D11Device *)gs_get_device_obj();
    ID3D11DeviceContext *d3dCtx = nullptr;
    d3dDev->GetImmediateContext(&d3dCtx);
    d3dCtx->CopyResource(f->sharedTex, srcTex);
    d3dCtx->Flush();
    d3dCtx->Release();

    // Step 4: CUDA interop
    cudaSetDevice(0);
    cudaFree(0);

    cudaError_t ce = cudaGraphicsMapResources(1, &f->cudaRes);
    if (ce != cudaSuccess) {
        blog(LOG_WARNING, "[YOLO] map fail: %s", cudaGetErrorString(ce)); return;
    }
    cudaArray_t arr = nullptr;
    cudaGraphicsSubResourceGetMappedArray(&arr, f->cudaRes, 0, 0);

    // GPU copy: 640x640 BGRA from CUDA array → d_bgra
    ce = cudaMemcpy2DFromArrayAsync(f->d_bgra, kInputW * 4, arr, 0, 0,
                                     kInputW * 4, kInputH,
                                     cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &f->cudaRes);

    if (ce != cudaSuccess) {
        blog(LOG_WARNING, "[YOLO] memcpy fail: %s", cudaGetErrorString(ce)); return;
    }

    cudaDeviceSynchronize();

    // Debug: dump the cropped input frame
    debugDumpPreprocessInput((const uint8_t *)f->d_bgra);

    // Launch NVRTC FP32 preprocess kernel
    {
        int texW = kInputW, texH = kInputH, c0 = 0;
        void *args[] = { &f->d_bgra, &texW, &texH, &f->d_input, &c0, &c0 };
        dim3 block(32, 8), grid(20, 80);
        cuLaunchKernel(f->cuKernel, grid.x, grid.y, 1,
                       block.x, block.y, 1, 0, nullptr, args, nullptr);
    }
    cudaDeviceSynchronize();

    // Step 5: TRT inference
    cudaEventRecord(f->evStart);
    bool ok = f->execCtx->enqueueV3(nullptr);
    if (!ok) {
        blog(LOG_WARNING, "[YOLO] enqueueV3 failed"); return;
    }
    cudaEventRecord(f->evStop);
    cudaEventSynchronize(f->evStop);
    cudaEventElapsedTime(&f->gpuLatencyMs, f->evStart, f->evStop);

    // Step 6: Readback detections
    std::vector<float> output(f->outputBytes / sizeof(float));
    cudaMemcpy(output.data(), f->d_output, f->outputBytes, cudaMemcpyDeviceToHost);

    // Step 7: Postprocess
    f->detections = postprocess(output.data(), w, h, cw, ch, cx, cy, f->confThreshold);
    f->inferCount++;

    if (f->inferCount % 120 == 1)
        blog(LOG_INFO, "[YOLO] #%d: %d dets | GPU %.2fms | outputBytes=%zu",
             f->inferCount, (int)f->detections.size(), f->gpuLatencyMs, f->outputBytes);

    // Step 8: Aim assist — priority: head (cls=1) > body upper 20% (cls=0)
    if (f->aimAssistEnabled && f->mouse.IsLoaded() && !f->detections.empty()) {
        const Detection *head = nullptr, *body = nullptr;
        for (const auto &d : f->detections) {
            if (d.cls == 1 && (!head || d.conf > head->conf)) head = &d;
            if (d.cls == 0 && (!body || d.conf > body->conf)) body = &d;
        }

        float aimX = 0, aimY = 0;
        const char *what = "none";

        if (head) {
            aimX = (head->x1 + head->x2) * 0.5f;
            aimY = (head->y1 + head->y2) * 0.5f;
            what = "head";
        } else if (body) {
            aimX = (body->x1 + body->x2) * 0.5f;
            aimY = body->y1 + (body->y2 - body->y1) * 0.10f; // upper 20% → center at 10% from top
            what = "body->head";
        }

        if (head || body) {
            float scx = w * 0.5f, scy = h * 0.5f;
            int dx = (int)((aimX - scx) * f->aimSensitivity);
            int dy = (int)((aimY - scy) * f->aimSensitivity);
            if (dx != 0 || dy != 0) f->mouse.MoveRelative(dx, dy);
            if (f->inferCount % 120 == 1)
                blog(LOG_INFO, "[YOLO] aim[%s]: (%.0f,%.0f) -> (%d,%d) conf=%.2f",
                     what, aimX, aimY, dx, dy,
                     head ? head->conf : body->conf);
        }
    }

    // Step 9: Draw detection boxes
    if (f->detections.empty()) return;

    gs_effect_t *solid = obs_get_base_effect(OBS_EFFECT_SOLID);
    gs_eparam_t *cp = gs_effect_get_param_by_name(solid, "color");
    struct vec4 green; vec4_set(&green, 0.0f, 1.0f, 0.0f, 1.0f);
    gs_effect_set_vec4(cp, &green);

    gs_matrix_push();
    gs_matrix_identity();
    gs_ortho(0.0f, (float)w, 0.0f, (float)h, -1.0f, 1.0f);
    float t = 2.0f;

    while (gs_effect_loop(solid, "Solid")) {
        for (const auto &d : f->detections) {
            float x1 = d.x1, y1 = d.y1, x2 = d.x2, y2 = d.y2;
            auto drawRect = [&](float l, float r, float t2, float b) {
                gs_render_start(false);
                gs_vertex2f(l, t2); gs_vertex2f(r, t2);
                gs_vertex2f(r, b); gs_vertex2f(l, b);
                gs_render_stop(GS_TRISTRIP);
            };
            drawRect(x1, x2, y1, y1 + t);
            drawRect(x1, x2, y2 - t, y2);
            drawRect(x1, x1 + t, y1, y2);
            drawRect(x2 - t, x2, y1, y2);
        }
    }
    gs_matrix_pop();
}

// ----------------------------------------------------------------
static obs_properties_t *filter_get_properties(void *) {
    obs_properties_t *props = obs_properties_create();
    obs_properties_add_path(props, "engine_path", "Engine Path",
                            OBS_PATH_FILE, "TensorRT Engine (*.engine)", nullptr);
    obs_properties_add_float_slider(props, "conf_threshold", "Confidence Threshold",
                                    0.0f, 1.0f, 0.05f);
    obs_properties_add_float_slider(props, "aim_sensitivity", "Aim Smoothing",
                                    0.01f, 1.0f, 0.01f);
    obs_properties_add_bool(props, "aim_assist", "Enable Aim Assist");
    return props;
}
static void filter_get_defaults(obs_data_t *settings) {
    obs_data_set_default_string(settings, "engine_path",
        "D:/code/obs-plugintemplate/data/best_trt.engine");
    obs_data_set_default_double(settings, "conf_threshold", 0.25);
    obs_data_set_default_double(settings, "aim_sensitivity", 0.15);
    obs_data_set_default_bool(settings, "aim_assist", false);
}

static struct obs_source_info yolo_filter = {
    .id = "yolo_trt_filter", .type = OBS_SOURCE_TYPE_FILTER, .output_flags = OBS_SOURCE_VIDEO,
    .get_name = filter_get_name, .create = filter_create, .destroy = filter_destroy,
    .get_defaults = filter_get_defaults, .get_properties = filter_get_properties,
    .update = filter_update, .video_tick = filter_video_tick, .video_render = filter_video_render,
};

bool obs_module_load(void) {
    obs_log(LOG_INFO, "plugin loaded successfully (version %s)", PLUGIN_VERSION);
    obs_register_source(&yolo_filter);
    return true;
}
void obs_module_unload(void) { obs_log(LOG_INFO, "plugin unloaded"); }
