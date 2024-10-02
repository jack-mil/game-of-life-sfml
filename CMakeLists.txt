# Minimum CMake version required
cmake_minimum_required(VERSION 3.26)

# Project Name
project(gol)

# Set the C++ standard
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenMP
# find_package(OpenMP REQUIRED)

# Static linking wasn't wuite working right
# Laser's aren't visible, no errors tho... strange
# option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
# set(SFML_STATIC_LIBRARIES TRUE)
option(SFML_BUILD_AUDIO FALSE)

# Fetch the SFML source code dependencies from github
include(FetchContent)
FetchContent_Declare(SFML
GIT_REPOSITORY https://github.com/SFML/SFML.git
GIT_TAG 2.6.x
GIT_SHALLOW ON
EXCLUDE_FROM_ALL
SYSTEM)
FetchContent_MakeAvailable(SFML)

# Add the executable
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE sfml-graphics)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_options(${PROJECT_NAME} PRIVATE
                                        -Wall
                                        -Wsign-conversion
                                        -Wconversion
                                        -Wextra
                                        # -Werror
                                        -pedantic
                                        -pedantic-errors)

# Link OpenMP libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(my_program PUBLIC OpenMP::OpenMP_CXX)
endif()
