# Определяем опции
option(TEST_CODE "Build TEST program" OFF)
option(WITH_MULTICORE "Build multicore version" ON)
option(WITH_OPENCL "Build OpenCL version" ON)
option(IS_MODULAR "Build modular version" ON)

# Устанавливаем пути
find_package(Threads REQUIRED)

message(STATUS "=== USING STANDARD OPENCV ===")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Добавляем директории с модулями
set(SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CONFIG_DIR ${SOURCES_DIR}/config)
set(DATA_DIR ${SOURCES_DIR}/data)
set(FILTERS_DIR ${SOURCES_DIR}/filters)
set(GPU_DIR ${SOURCES_DIR}/gpu)
set(STABILIZER_DIR ${SOURCES_DIR}/stabilizer)
set(SYNC_DIR ${SOURCES_DIR}/sync)
set(UTILS_DIR ${SOURCES_DIR}/utils)
set(SANDBOX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sandbox)

# ========================= ВЫБОР ЦЕЛЕВОГО ПРИЛОЖЕНИЯ =========================
# Приоритет порядка:
# 1) WITH_MULTICORE AND WITH_OPENCL -> src/mainMultiCoreOpenCL.cpp
# 2) WITH_MULTICORE AND WITH_OPENCL AND IS_MODULAR -> src/main.cpp (модули)
# 3) TEST_CODE -> sandbox/mainTest.cpp
# 4) Остальные случаи -> sandbox/main.cpp

if(TEST_CODE)
    # 3) Тестовые примеры из sandbox
    add_executable(myapp_test 
        ${SANDBOX_DIR}/mainTest.cpp
        ${SANDBOX_DIR}/sandbox.cpp
        ${SANDBOX_DIR}/sandboxFunctionsDev.cpp
        ${SANDBOX_DIR}/basicFunctions.cpp
        ${SANDBOX_DIR}/stabilizationFunctions.cpp
        ${SANDBOX_DIR}/wienerFilter.cpp
    )
    
    target_include_directories(myapp_test PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CONFIG_DIR}
        ${FILTERS_DIR}
        ${GPU_DIR}
        ${STABILIZER_DIR}
        ${SYNC_DIR}
        ${UTILS_DIR}
        ${SANDBOX_DIR}
    )
    
    message(STATUS "=== BUILDING TEST PROGRAM (sandbox/mainTest.cpp) ===")
    target_link_libraries(myapp_test ${OpenCV_LIBS} Threads::Threads)

elseif(IS_MODULAR)
    # 2) Модульная версия на основе src/main.cpp
    add_executable(myapp_modular 
        main.cpp
        utils/helpers.cpp
        stabilizer/video_stabilizer.cpp
        gpu/warp_affine.cpp
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
    
    message(STATUS "=== BUILDING MODULAR VERSION (src/main.cpp) ===")
    target_link_libraries(myapp_modular ${OpenCV_LIBS} Threads::Threads)

elseif(WITH_MULTICORE AND WITH_OPENCL)
    # 1) Версия на основе mainMultiCoreOpenCL.cpp
    add_executable(mainMultiCoreOpenCL_app
        mainMultiCoreOpenCL.cpp
    )
    
    target_include_directories(mainMultiCoreOpenCL_app PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CONFIG_DIR}
        ${FILTERS_DIR}
        ${GPU_DIR}
        ${STABILIZER_DIR}
        ${SYNC_DIR}
        ${UTILS_DIR}
    )
    
    message(STATUS "=== BUILDING OPENCL VERSION (src/mainMultiCoreOpenCL.cpp) ===")
    target_link_libraries(mainMultiCoreOpenCL_app ${OpenCV_LIBS} Threads::Threads)

else()
    # 4) Остальные случаи - sandbox/main.cpp
    add_executable(myapp_sandbox 
        ${SANDBOX_DIR}/main.cpp
        ${SANDBOX_DIR}/sandbox.cpp
        ${SANDBOX_DIR}/sandboxFunctionsDev.cpp
        ${SANDBOX_DIR}/basicFunctions.cpp
        ${SANDBOX_DIR}/stabilizationFunctions.cpp
        ${SANDBOX_DIR}/wienerFilter.cpp
    )
    
    target_include_directories(myapp_sandbox PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CONFIG_DIR}
        ${FILTERS_DIR}
        ${GPU_DIR}
        ${STABILIZER_DIR}
        ${SYNC_DIR}
        ${UTILS_DIR}
        ${SANDBOX_DIR}
    )
    
    message(STATUS "=== BUILDING SANDBOX VERSION (sandbox/main.cpp) ===")
    target_link_libraries(myapp_sandbox ${OpenCV_LIBS} Threads::Threads)
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
