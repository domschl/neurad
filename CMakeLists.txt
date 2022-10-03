cmake_minimum_required(VERSION 3.0.0)
project(neurad VERSION 0.1.0)

include(CTest)
enable_testing()

# Snatched from Fuchsia:
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build Type" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Set a value type to properly display CMAKE_BUILD_TYPE on GUI if the
# value type is "UNINITIALIZED".
GET_PROPERTY(cached_type CACHE CMAKE_BUILD_TYPE PROPERTY TYPE)

IF("${cached_type}" STREQUAL "UNINITIALIZED")
    SET(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Build Type" FORCE)
ENDIF("${cached_type}" STREQUAL "UNINITIALIZED")

# Check the Build Type.
IF(NOT "${CMAKE_BUILD_TYPE}"
    MATCHES "^(Debug|Release|RelWithDebInfo|MinSizeRel)\$")
    MESSAGE(FATAL_ERROR
        "Unknown keyword for CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}\n"
        "Acceptable keywords: Debug,Release,RelWithDebInfo,MinSizeRel")
ENDIF(NOT "${CMAKE_BUILD_TYPE}"
    MATCHES "^(Debug|Release|RelWithDebInfo|MinSizeRel)\$")

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
        include_directories("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/")
        link_directories(${BLAS_LIBRARIES})
    else()
        link_directories(${BLAS_LIBRARIES})
    endif(APPLE)

    message(STATUS "BLAS " ${BLA_VENDOR} " found")
endif(BLAS_FOUND)

# if(APPLE)
# target_link_libraries(neurad LINK_PUBLIC ${BLAS_LIBRARIES})
# else()
# target_link_libraries(neurad LINK_PUBLIC ${BLAS_LIBRARIES} cblas)
# endif()
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

include_directories("core")

add_subdirectory("test")
add_subdirectory("bench")

# add_executable(neurad main.cpp)
