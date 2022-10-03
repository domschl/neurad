add_executable(bench_neurad bench_neurad.cpp)

if(Apple)
    target_link_libraries(bench_neurad LINK_PUBLIC ${BLAS_LIBRARIES})
else()
    target_link_libraries(bench_neurad LINK_PUBLIC ${BLAS_LIBRARIES} cblas)
endif()