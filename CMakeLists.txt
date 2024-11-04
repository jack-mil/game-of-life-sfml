# Minimum CMake version required
cmake_minimum_required(VERSION 3.26)

# Project Name (Game Of Life)
project(gol)

# Build for release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

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

# Add the header-only libraries
add_library(third-party INTERFACE)

# SYSTEM keyword prevents warnings from external libraries
target_include_directories(third-party SYSTEM INTERFACE "third-party/")

# Add the executable target
add_executable(${PROJECT_NAME}
                src/main.cpp 
                src/App.cpp 
                src/Life.cpp
)

# Include my headers
target_include_directories(${PROJECT_NAME} PRIVATE include/)

# link dependency libraries (static SFML)
target_link_libraries(${PROJECT_NAME} PRIVATE third-party sfml-graphics)

# require c++20 features
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# enable compiler warnings
set(DEBUG_OPTIONS -Wall -Wextra -Werror -pedantic -pedantic-errors)
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")
