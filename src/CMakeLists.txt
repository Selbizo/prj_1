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

#banana PI config v2
set(OpenCV_DIR "/home/bananapi/opencv4120/opencv/build")
find_package(OpenCV REQUIRED)
add_executable( run_v2 main2.cpp )
include_directories(${OpenCV_INCLUDE_DIRS} )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
message(${PROJECT_SOURCE_DIR})
target_link_libraries( run_v2 ${OpenCV_LIBS})