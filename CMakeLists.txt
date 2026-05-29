cmake_minimum_required(VERSION 3.28...3.30)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/common/bootstrap.cmake" NO_POLICY_SCOPE)

project(${_name} VERSION ${_version} LANGUAGES C CXX)

option(ENABLE_FRONTEND_API "Use obs-frontend-api for UI functionality" OFF)
option(ENABLE_QT "Use Qt functionality" OFF)

include(compilerconfig)
include(defaults)
include(helpers)

add_library(${CMAKE_PROJECT_NAME} MODULE)

find_package(libobs REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::libobs)

# Paths from train_custom_dataset.md
set(TRT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.16.1.11")
set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${TRT_ROOT}/include
    ${CUDA_ROOT}/include
)

target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${TRT_ROOT}/lib
    ${CUDA_ROOT}/lib/x64
)

if(ENABLE_FRONTEND_API)
  find_package(obs-frontend-api REQUIRED)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::obs-frontend-api)
endif()

if(ENABLE_QT)
  find_package(Qt6 COMPONENTS Widgets Core)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Qt6::Core Qt6::Widgets)
  target_compile_options(
    ${CMAKE_PROJECT_NAME}
    PRIVATE $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-quoted-include-in-framework-header -Wno-comma>
  )
  set_target_properties(
    ${CMAKE_PROJECT_NAME}
    PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON
  )
endif()

target_sources(${CMAKE_PROJECT_NAME} PRIVATE src/plugin-main.cpp src/debug_utils.cpp)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    _CRT_SECURE_NO_WARNINGS
    NOMINMAX
)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE /utf-8 /wd4996)
target_link_options(${CMAKE_PROJECT_NAME} PRIVATE /NODEFAULTLIB:LIBCMT /NODEFAULTLIB:LIBCMTD)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
    nvinfer_10
    nvinfer_plugin_10
    nvonnxparser_10
    cuda
    cudart
    nvrtc
)

set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${_name})
