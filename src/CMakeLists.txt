# Определяем опции
option(TEST_CODE "Build TEST program" OFF)
option(WITH_MULTICORE "Build multicore version" ON)
option(WITH_OPENCL "Build OpenCL version" ON)

# Устанавливаем пути
find_package(Threads REQUIRED)

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
        message(STATUS "=== BUILDING OPENCL VERSION ===")
        target_link_libraries(myapp_opencl ${OpenCV_LIBS} Threads::Threads)
    elseif(WITH_MULTICORE)
        add_executable(myapp mainMultiCore.cpp)
        message(STATUS "=== BUILDING MULTICORE VERSION ===")
        target_link_libraries(myapp ${OpenCV_LIBS} Threads::Threads)
    else()
        add_executable(myapp main.cpp)
        message(STATUS "=== BUILDING SINGLECORE VERSION ===")
        target_link_libraries(myapp ${OpenCV_LIBS} Threads::Threads basicFunctions stabilizationFunctions)
    endif()
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})