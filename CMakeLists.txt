cmake_minimum_required(VERSION 3.20)

project(HMLR LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Armadillo REQUIRED)

add_executable(HMLR_test
    tests/HMLR_test.cpp
    src/HMLR.cpp
    src/PCBall.cpp
)

target_compile_options(HMLR_test PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-mcpu=native>
)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

if(ipo_supported)
    set_property(TARGET HMLR_test
        PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()
target_compile_features(HMLR_test PRIVATE cxx_std_23)

target_include_directories(HMLR_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/test"
    ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(HMLR_test PRIVATE
    ${ARMADILLO_LIBRARIES}
)
