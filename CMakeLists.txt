cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(QuantInferenceEngine)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(quant_engine main.cpp)
target_link_libraries(quant_engine "${TORCH_LIBRARIES}")
set_property(TARGET quant_engine PROPERTY CXX_STANDARD 17)