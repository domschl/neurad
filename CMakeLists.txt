cmake_minimum_required(VERSION 3.0.0)
project(neurad VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(BLAS REQUIRED)

if(BLAS_FOUND)
    include_directories(${BLAS_INCLUDE_DIRS})
    link_directories(${BLAS_LINK_DIRS})
    message(STATUS "BLAS found.")
endif(BLAS_FOUND)

add_executable(neurad main.cpp)
target_link_libraries(neurad LINK_PUBLIC ${BLAS_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
