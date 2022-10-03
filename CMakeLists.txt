cmake_minimum_required(VERSION 3.0.0)
project(neurad VERSION 0.1.0)

include(CTest)
enable_testing()

# Build type needs to be set as parameter to CMAKE: -DCMAKE_BUILD_TYPE=xxxx
# set(CMAKE_BUILD_TYPE Release)
# If the user specifies -DCMAKE_BUILD_TYPE on the command line, take their definition
# and dump it in the cache along with proper documentation, otherwise set CMAKE_BUILD_TYPE
# to Debug prior to calling PROJECT()
IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of `build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE()
   SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()

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
    SET(BLA_VENDOR "Apple" CACHE STRING "Set BLAS vendor (see FindBLAS) e.g. OpenBLAS, Apple, Generic")
else()
    SET(BLA_VENDOR "OpenBLAS" CACHE STRING "Set BLAS vendor (see FindBLAS) e.g. OpenBLAS, Apple, Generic")
    SET(CMAKE_C_FLAGS -fopenmp)
endif()

find_package(BLAS REQUIRED)

if(BLAS_FOUND)
    if(APPLE)
        # That looks well organised: (used for editor code completion)
        include_directories("/Library/Developer/CommandLineTools/SDKs/MacOSX13.0.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/")
        link_directories(${BLAS_LIBRARIES})
    else()
        link_directories(${BLAS_LIBRARIES})
    endif(APPLE)

    message(STATUS "BLAS " ${BLA_VENDOR} " found")
endif(BLAS_FOUND)

add_executable(neurad main.cpp)

if(APPLE)
    target_link_libraries(neurad LINK_PUBLIC ${BLAS_LIBRARIES})
else()
    target_link_libraries(neurad LINK_PUBLIC ${BLAS_LIBRARIES} cblas)
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
