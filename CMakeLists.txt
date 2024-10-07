# Minimum CMake version required
cmake_minimum_required(VERSION 3.26)

# Project Name (Game Of Life)
project(gol)

# Set the C++ standard
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenMP
# find_package(OpenMP REQUIRED)

# Link to SFML statically
set(BUILD_SHARED_LIBS FALSE)
# Disable building audio
set(SFML_BUILD_AUDIO FALSE)

# Fetch the SFML source code dependencies from github
include(FetchContent)
    FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 2.6.x
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM
)
FetchContent_MakeAvailable(SFML)

# Add the command line argument parser header-only library
add_library(clipp INTERFACE)

# SYSTEM keyword prevents warnings from external libraries
target_include_directories(clipp SYSTEM INTERFACE "third-party/")

# Add the executable
add_executable(${PROJECT_NAME} src/main.cpp)

# link libraries (static SFML)
target_link_libraries(${PROJECT_NAME} PRIVATE clipp sfml-graphics)

# Link OpenMP libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()

# require c++20 features
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)


# enable compiler warnings
target_compile_options(${PROJECT_NAME} PRIVATE
                                        -Wall
                                        # -Wsign-conversion
                                        # -Wconversion
                                        -Wextra
                                        -Werror
                                        -pedantic
                                        -pedantic-errors)

                                        