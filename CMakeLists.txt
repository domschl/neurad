cmake_minimum_required(VERSION 3.0.0)
project(neurad VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

option(SINGLE_PRECISION_FLOAT "Use float 32-bit precision (OFF=double)" ON)

if(SINGLE_PRECISION_FLOAT)
    add_definitions(-DUSE_SINGLE_PRECISION_FLOAT)
    message(STATUS "Single precision")
else()
    add_definitions(-DUSE_DOUBLE_PRECISION_FLOAT)
    message(STATUS "Double precision")
endif()

if(APPLE)
    message(STATUS "Apple platform")
    include_directories("/usr/local/include")
    include_directories("/opt/homebrew/include")
    SET(BLA_VENDOR "Apple" CACHE STRING "Set BLAS vendor (see FindBLAS) e.g. OpenBlas, Apple, Generic")
else()
    SET(BLA_VENDOR "" CACHE STRING "Set BLAS vendor (see FindBLAS) e.g. OpenBlas, Apple, Generic")
endif()

find_package(BLAS REQUIRED)

if(BLAS_FOUND)
    if(APPLE)
        # That looks well organised:
        include_directories("/Library/Developer/CommandLineTools/SDKs/MacOSX13.0.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/")
    endif(APPLE)

    link_directories(${BLAS_LINK_DIRS})
    message(STATUS "BLAS " ${BLA_VENDOR} " found, INC:" ${BLAS_INCLUDE_DIRS})
endif(BLAS_FOUND)

add_executable(neurad main.cpp)
target_link_libraries(neurad LINK_PUBLIC ${BLAS_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)