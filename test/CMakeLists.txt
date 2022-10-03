add_executable(test_neurad test_neurad.cpp)

if(Apple)
    target_link_libraries(test_neurad LINK_PUBLIC ${BLAS_LIBRARIES})
else()
    target_link_libraries(test_neurad LINK_PUBLIC ${BLAS_LIBRARIES} cblas)
endif()