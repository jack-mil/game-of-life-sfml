/*
Author: Jackson Miller
Class: ECE6122 A
Last Date Modified: 2024-10-10

Description:
-- Lab 2 --
Game of Life implemented with a choice of several multithreading techniques

Utilizes command line argument parsing with a third-party library
(ain't no way I'm implementing that myself (hard))

Look at App.cpp for the window/graphics/game loop stuff
and at Life.cpp for the nitty-gritty Life implementation
*/

#include <iostream>

// Compiled from source using CMake FetchContent
#include <SFML/Graphics.hpp>

// Command line argument parsing
// https://github.com/muellan/clipp
#include <clipp.h>

#include "App.hpp" // App, Mode

/** Parse command line arguments and start the application */
int main(int argc, char* argv[])
{
    // Runtime parameters and defaults
    Mode   mode    = Mode::Threads;
    bool   no_gui  = false;
    uint   threads = 8;
    size_t size    = 5;
    size_t width   = 800;
    size_t height  = 600;

    { // clang-format off

    // just for this block
    using namespace clipp;
    using namespace std;

    // Define command line arguments using clipp
    auto cli = (
        (option("-x", "--width" )  & integer("WIDTH" ).set(width )).doc("Width of the window (default=" + to_string(width)          + ")" ),
        (option("-y", "--height")  & integer("HEIGHT").set(height)).doc("Height of the window (default=" + to_string(height)        + ")" ),
        (option("-c", "--size")    & integer("SIZE"  ).set(size  )).doc("Size in pixels of each cell (default=" + to_string(size)   + ")" ),
        (option("-n", "--threads") & integer("COUNT" ).set(threads)).doc("How many threads to use (>2) (default=" + to_string(threads) + ")" ),
        (option("-t", "--mode")    & one_of(
                                    required("SEQ" ).set(mode, Mode::Sequential),
                                    required("THRD").set(mode, Mode::Threads),
                                    required("OMP" ).set(mode, Mode::OpenMP) ) ).doc("Type of parallelism to use (default: std::thread)"),
        option("-d","--no-gui").set(no_gui).doc("Only run performance timings.")
    );

    // If any errors parsing, print help and exit
    if(!parse(argc, argv, cli)) {

        auto fmt = doc_formatting{}
                    .first_column(2)
                    .doc_column(31)
                    .max_flags_per_param_in_usage(4);

        std::cout << "ECE 6122 Lab 2\n\n"
                    << "Usage:\n"
                    << usage_lines(cli, argv[0], fmt)
                    << "\n\nOptions:\n"
                    << documentation(cli, fmt)
                    << std::endl;

        return EXIT_FAILURE;
    }

    } // clang-format on

    // cli parameters available to use here

    // create the app with command line arguments (or defaults if not specified)
    App app{width, height, size, mode, threads, no_gui};

    // run the SFML loop
    app.run();

    return EXIT_SUCCESS;
}
