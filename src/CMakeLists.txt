#banana PI config v1
# set(OpenCV_DIR "/media/pi/sd30gb/opencv340/opencv/build")
set(OpenCV_DIR "/media/pi/sd30gb/opencv4120/opencv/build")
find_package(OpenCV REQUIRED)

add_library(basicFunctions SHARED "basicFunctions.cpp")
add_library(stabilizationFunctions SHARED "stabilizationFunctions.cpp")
#add_library(wienerFilter SHARED "wienerFilter.cpp")
target_link_libraries(basicFunctions ${OpenCV_LIBS})
target_link_libraries(stabilizationFunctions ${OpenCV_LIBS})
#target_link_libraries(wienerFilter ${OpenCV_LIBS})

add_executable( run main.cpp )

include_directories(${OpenCV_INCLUDE_DIRS} )

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
message(${PROJECT_SOURCE_DIR})
# target_link_libraries( run ${OpenCV_LIBS} basicFunctions stabilizationFunctions wienerFilter)
target_link_libraries( run ${OpenCV_LIBS} basicFunctions stabilizationFunctions )


#~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# find_package(OpenCL REQUIRED)

# OpenCV
# set(OpenCV_DIR "/media/pi/sd30gb/opencv340/opencv/build")
# find_package(OpenCV REQUIRED)

# add_executable(run main.cpp)
# target_link_libraries(run ${OpenCV_LIBS})