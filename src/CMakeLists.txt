# Определяем опции
option(TEST_CODE "Build TEST program" OFF)
option(WITH_MULTICORE "Build multicore version" ON)
option(WITH_OPENCL "Build OpenCL version" ON)

# Устанавливаем пути
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# Добавляем флаги для NEON оптимизации на ARM
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*")
    message(STATUS "=== ARM Architecture Detected ===")
    # Включаем NEON инструкции
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -march=armv7-a")
    message(STATUS "NEON flags enabled: -mfpu=neon -march=armv7-a")
endif()

message(STATUS "=== USING STANDARD OPENCV ===")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Выбор целевого приложения
if(TEST_CODE)
    add_executable(myapp_test mainTest.cpp)
    message(STATUS "=== BUILDING TEST PROGRAM ===")
    target_link_libraries(myapp_test ${OpenCV_LIBS} Threads::Threads)
else()
    if(WITH_MULTICORE AND WITH_OPENCL)
        #add_executable(myapp_opencl mainOpenCL.cpp)
        add_executable(myapp_opencl mainMultiCoreOpenCL.cpp)
        message(STATUS "=== BUILDING MULTICORE OPENCL VERSION WITH NEON OPTIMIZATION ===")
        target_link_libraries(myapp_opencl ${OpenCV_LIBS} Threads::Threads OpenMP::OpenMP_CXX)
    elseif(WITH_MULTICORE)
        add_executable(myapp_multicore mainMultiCore.cpp)
        message(STATUS "=== BUILDING MULTICORE VERSION WITH NEON OPTIMIZATION ===")
        target_link_libraries(myapp_multicore ${OpenCV_LIBS} Threads::Threads OpenMP::OpenMP_CXX)
    else()
        add_executable(myapp_singlecore main.cpp)
        message(STATUS "=== BUILDING SINGLECORE VERSION ===")
        target_link_libraries(myapp_singlecore ${OpenCV_LIBS} Threads::Threads basicFunctions stabilizationFunctions)
    endif()
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})