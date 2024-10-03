/*
Author: Jackson Miller
Class: ECE6122 A
Last Date Modified: 2024-10-02

Description:
-- Lab 2 --
Game of Life implemented with a choice of several multithreading techniques
*/

#include <iostream>
#include <random>
#include <SFML/Graphics.hpp>
#include <vector>

// Command line argument parsing
// https://github.com/muellan/clipp
#include "clipp.h"

// Type alias to save space
using Grid = std::vector<std::vector<bool>>;

/**
 * Initialize each cell of the grid with a random true/false value
 *
 * Makes use of the bernoulli distribution and hardware random number generator.
 * @param grid reference to 2D Vector<bool>, modified in place.
 */
void seedRandomGrid(Grid& grid)
{
    std::default_random_engine  gen(std::random_device{}());
    std::bernoulli_distribution coin_flip(0.5);
    for (auto& col : grid)
    {
        for (auto cell : col)
        {
            cell = coin_flip(gen);
        }
    }
}

/**
 * Find the number of live cells around a point x,y on the grid
 *
 * @param grid world to check
 * @param x column pos
 * @param y row pos
 * @return uint number of neighbors
 */
uint countNeighbors(const Grid& grid, Grid::size_type x, Grid::size_type y)
{
    uint count = 0;
    for (int i = -1; i <= 1; ++i)
    {
        for (int j = -1; j <= 1; ++j)
        {
            if (i == 0 && j == 0)
            {
                continue;
            }
            Grid::size_type nx = (x + i + grid.size()) % grid.size();
            Grid::size_type ny = (y + j + grid[x].size()) % grid[x].size();

            count += static_cast<uint>(grid[nx][ny]); // interpret bool as 1/0
        }
    }
    return count;
}

/**
 * Create SFML rectangles for every live cell
 *
 * @param grid world to draw
 * @param cellSize square pixels of each cell
 * @param window window to draw to
 */
void drawGrid(const Grid& grid, const int cellSize, sf::RenderTarget& window)
{
    for (size_t x = 0; const auto& col : grid)
    {
        for (size_t y = 0; bool cell : col)
        {
            if (cell)
            {
                sf::RectangleShape rect(sf::Vector2f(cellSize, cellSize));
                rect.setPosition(x * cellSize, y * cellSize);
                rect.setFillColor(sf::Color::White);
                window.draw(rect);
            }
            y++;
        }
        x++;
    }
}

/**
 * Given the current state, write the next state to `next`,
 * according to the classic Game of Life rules
 * @param grid
 * @param next
 */
void updateGrid(const Grid& grid, Grid& next)
{
    for (size_t x = 0; x < grid.size(); ++x)
    {
        for (size_t y = 0; y < grid[x].size(); ++y)
        {
            uint neighbors = countNeighbors(grid, x, y);

            if (grid[x][y])
            {
                if (neighbors < 2 || neighbors > 3)
                {
                    next[x][y] = false; // Cell dies
                }
                else
                {
                    next[x][y] = true; // Continues to live
                }
            }
            else
            {
                if (neighbors == 3)
                {
                    next[x][y] = true; // Cell becomes alive
                }
                else
                {
                    next[x][y] = false; // Remains dead
                }
            }
        }
    }
}

/** Convenience function to center the window on the desktop */
void setupWindow(uint width, uint height, sf::RenderWindow& window)
{
    window.create(sf::VideoMode(width, height), "Game of Life - Jackson Miller");
    // set some OS window options
    window.setMouseCursorVisible(false);
    window.setVerticalSyncEnabled(false);
    window.setFramerateLimit(120);

    // place the window in the center of the desktop
    const auto& desktop = sf::VideoMode::getDesktopMode();
    const auto  xpos    = (desktop.width / 2u) - (window.getSize().x / 2u);
    const auto  ypos    = (desktop.height / 2u) - (window.getSize().y / 2u);
    window.setPosition(sf::Vector2i(static_cast<int>(xpos), static_cast<int>(ypos)));
}

/** Handle events to close the window with ESC or X button */
void handleWindowEvents(sf::RenderWindow& window)
{
    sf::Event event;
    while (window.pollEvent(event))
    {
        if (event.type == sf::Event::Closed)
        {
            window.close();
        }
        else if (event.type == sf::Event::KeyPressed)
        {
            if (event.key.code == sf::Keyboard::Escape || event.key.code == sf::Keyboard::Return)
            {
                window.close();
            }
        }
    }
}

int main(int argc, char* argv[])
{
    enum class Mode { Sequencial, Threads, OpenMp };

    // Runtime parameters and defaults
    Mode mode   = Mode::Sequencial;
    int  count  = 8;
    int  size   = 5;
    int  width  = 800;
    int  height = 600;

    { // clang-format off
    using namespace clipp;
    using namespace std;

    // Define command line arguments using clipp
    auto cli = (
        (option("-x", "--width" )  & integer("WIDTH" ).set(width )).doc("Width of the window (default=" + to_string(width)          + ")" ),
        (option("-y", "--height")  & integer("HEIGHT").set(height)).doc("Height of the window (default=" + to_string(height)        + ")" ),
        (option("-c", "--size")    & integer("SIZE"  ).set(size  )).doc("Size in pixels of each cell (default=" + to_string(size)   + ")" ),
        (option("-n", "--threads") & integer("COUNT" ).set(count )).doc("How many threads to use (>2) (default=" + to_string(count) + ")" ),
        (option("-t", "--mode")    & one_of(
                                    required("SEQ" ).set(mode, Mode::Sequencial),
                                    required("THRD").set(mode, Mode::Threads),
                                    required("OMP" ).set(mode, Mode::OpenMp) ) ).doc("Type of parallelism to use (default: Sequential)")
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
    };
    } // clang-format on

    // parameters available to use here...

    const int grid_width  = width / size;
    const int grid_height = height / size;

    sf::RenderWindow window;
    setupWindow(width, height, window);

    Grid grid_current(grid_width, std::vector<bool>(grid_height, false));
    Grid grid_next(grid_width, std::vector<bool>(grid_height, false));

    seedRandomGrid(grid_current);

    // use pointers because we want to swap without copying
    Grid* current = &grid_current;
    Grid* next    = &grid_next;

    /* Do 'game' loop */
    while (window.isOpen())
    {
        // check for close
        handleWindowEvents(window);

        // write next generation into next
        updateGrid(*current, *next);

        // make everything black
        window.clear();

        // draw live cells as white rectangles
        drawGrid(*current, size, window);

        // draw graphics to screen
        window.display();

        // swap pointers, ready for next iteration.
        std::swap(current, next);
    }

    return EXIT_SUCCESS;
}
