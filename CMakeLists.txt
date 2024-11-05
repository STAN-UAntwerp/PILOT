cmake_minimum_required(VERSION 3.12)
project(pilot_wrapper)

set(CMAKE_CXX_STANDARD 17)
set(pybind11_DIR /home/servot82/miniconda3/lib/python3.11/site-packages/pybind11/share/cmake/pybind11)

find_package(pybind11 REQUIRED)
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(carma CONFIG REQUIRED)

# Set the output directory for the compiled module
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/pilot)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

pybind11_add_module(cpilot pilot_wrapper.cpp tree.cpp)

target_link_libraries(cpilot PRIVATE carma::carma ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} pybind11::module)