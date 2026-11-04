cmake_minimum_required(VERSION 3.5)

find_package(Threads REQUIRED)

message(STATUS "=== USING STANDARD OPENCV ===")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# NEON-warp должен быть быстрым даже в Debug-сборке (иначе -O0 убивает весь
# смысл ручной векторизации). Основной проект при этом не трогаем.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/gpu/warp_affine_neon.cpp
        PROPERTIES COMPILE_OPTIONS "-O3;-DNDEBUG")
endif()

set(SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CONFIG_DIR ${SOURCES_DIR}/config)
set(FILTERS_DIR ${SOURCES_DIR}/filters)
set(GPU_DIR ${SOURCES_DIR}/gpu)
set(STABILIZER_DIR ${SOURCES_DIR}/stabilizer)
set(SYNC_DIR ${SOURCES_DIR}/sync)
set(UTILS_DIR ${SOURCES_DIR}/utils)

# Единственная цель проекта — MIPI/V4L2-версия стабилизатора.
add_executable(myapp_modular
    main.cpp
    utils/helpers.cpp
    stabilizer/video_stabilizer.cpp
    gpu/warp_affine.cpp
    gpu/warp_affine_neon.cpp
    filters/wiener_filter.cpp
)

target_include_directories(myapp_modular PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CONFIG_DIR}
    ${FILTERS_DIR}
    ${GPU_DIR}
    ${STABILIZER_DIR}
    ${SYNC_DIR}
    ${UTILS_DIR}
)

message(STATUS "=== BUILDING MIPI/V4L2 VERSION (src/main.cpp) ===")
target_link_libraries(myapp_modular ${OpenCV_LIBS} Threads::Threads)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
