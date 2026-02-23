cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(FisherKolmogorov src/FisherKolmogorov.cpp src/DiffusionNonLinear.cpp)
deal_ii_setup_target(FisherKolmogorov)
