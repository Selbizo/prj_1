cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 17)
project(Digital_Video_Stabilization)

# По умолчанию собираем с оптимизацией (Release = -O3 -DNDEBUG).
# Раньше проект собирался в Debug (-g, без -O) — это заметно замедляло код конвейера.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "=== CMAKE_BUILD_TYPE не задан, выбран Release (-O3) ===")
endif()

add_subdirectory(src)