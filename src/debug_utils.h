#pragma once
#include <cstdint>

// Set to true to enable debug frame dumping
inline constexpr bool kDebugDumpFrames = false;
// Save every Nth frame
inline constexpr int kDebugDumpInterval = 120;

// Dump the center-cropped preprocess input sent to TRT.
// Call this from filter_video_render after the CUDA crop memcpy, before kernel launch.
// bgra: device pointer to 640x640 BGRA buffer on GPU
void debugDumpPreprocessInput(const uint8_t *d_bgra);
