#banana PI config
add_library(basicFunctions SHARED "basicFunctions.cpp")
add_library(stabilizationFunctions SHARED "stabilizationFunctions.cpp")
add_library(wienerFilter SHARED "wienerFilter.cpp")
target_link_libraries(basicFunctions ${OpenCV_LIBS})
target_link_libraries(stabilizationFunctions ${OpenCV_LIBS})
target_link_libraries(wienerFilter ${OpenCV_LIBS})

add_executable( run main.cpp )

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} )

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
message(${PROJECT_SOURCE_DIR})
target_link_libraries( run ${OpenCV_LIBS} basicFunctions stabilizationFunctions wienerFilter)

