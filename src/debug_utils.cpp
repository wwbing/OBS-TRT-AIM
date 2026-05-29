#include "debug_utils.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include <vector>

static int g_dumpCount = 0;

void debugDumpPreprocessInput(const uint8_t *d_bgra) {
    if (!kDebugDumpFrames) return;
    if (g_dumpCount % kDebugDumpInterval != 0) { g_dumpCount++; return; }
    g_dumpCount++;

    // Heap allocation - stack would overflow OBS render thread
    std::vector<uint8_t> host(640 * 640 * 4);
    cudaError_t err = cudaMemcpy(host.data(), d_bgra, 640 * 640 * 4, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return;

    // BGRA → RGB (drop alpha, swap B/R)
    std::vector<uint8_t> rgb(640 * 640 * 3);
    for (int i = 0; i < 640 * 640; i++) {
        rgb[i*3+0] = host[i*4+2];
        rgb[i*3+1] = host[i*4+1];
        rgb[i*3+2] = host[i*4+0];
    }

    char path[256];
    snprintf(path, sizeof(path), "D:/code/obs-plugintemplate/data/debug_frame_%04d.jpg", g_dumpCount);
    stbi_write_jpg(path, 640, 640, 3, rgb.data(), 90);
}
