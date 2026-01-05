#banana PI config v1
# set(OpenCV_DIR "/home/bananapi/opencv4120/opencv/build")
# find_package(OpenCV REQUIRED)
# add_library(basicFunctions SHARED "basicFunctions.cpp")
# add_library(stabilizationFunctions SHARED "stabilizationFunctions.cpp")
# target_link_libraries(basicFunctions ${OpenCV_LIBS})
# target_link_libraries(stabilizationFunctions ${OpenCV_LIBS})
# add_executable( run main.cpp )
# include_directories(${OpenCV_INCLUDE_DIRS} )
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
# message(${PROJECT_SOURCE_DIR})
# target_link_libraries( run ${OpenCV_LIBS} basicFunctions stabilizationFunctions )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# #banana PI config v2
# set(OpenCV_DIR "/home/bananapi/opencv4120/opencv/build")
# set(OpenCV_DIR "/home/bananapi/opencv4120vulkan/opencv/build/")
# find_package(OpenCV REQUIRED)
# add_executable( run main2.cpp )
# include_directories(${OpenCV_INCLUDE_DIRS} )
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
# message(${PROJECT_SOURCE_DIR})
# target_link_libraries( run ${OpenCV_LIBS})



#banana PI config vulcan
# set(OpenCV_DIR "/home/bananapi/opencv4120vulkan/opencv/build/")
# find_package(OpenCV REQUIRED)
# add_executable( runMultiCore mainMultiCore.cpp )
# include_directories(${OpenCV_INCLUDE_DIRS} )
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
# message(${PROJECT_SOURCE_DIR})
# target_link_libraries( runMultiCore ${OpenCV_LIBS})


# Определяем опции
option(WITH_MULTICORE "Build multicore version" ON)
# option(WITH_MULTICORE "Build multicore version" OFF)

# option(TEST_CODE "Build test version" OFF)

# Устанавливаем пути

set(OpenCV_DIR "/home/bananapi/opencv4120vulkan/opencv/build")
message(STATUS "=== USING STANDARD OPENCV ===")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Выбор целевого приложения
# if(TEST_CODE)
#     add_executable(myapp mainTest.cpp)
#     message(STATUS "=== BUILDING MULTICORE VERSION ===")
# else()

if(WITH_MULTICORE)
add_executable(myapp mainMultiCore.cpp)
message(STATUS "=== BUILDING MULTICORE VERSION ===")
else()
add_executable(myapp mainMultiThread.cpp)
message(STATUS "=== BUILDING SINGLECORE VERSION ===")
endif()
# endif()

target_link_libraries(myapp ${OpenCV_LIBS})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})