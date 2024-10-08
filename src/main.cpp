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
#include <clipp.h>

// Simple tread pooling implementation
// https://github.com/alugowski/task-thread-pool
#include <task-thread-pool.hpp>

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
inline uint countNeighbors(const Grid& grid, size_t x, size_t y)
{
    uint count = 0;
    // This will probably be unrolled
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

inline void updateSingleCell(const Grid& current, Grid& next, size_t x, size_t y)
{
    uint neighbors = countNeighbors(current, x, y);

    if (current[x][y])
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
 * according to the classic Game of Life rules.
 * Uses regular sequential processing.
 * @param current
 * @param next
 */
void updateGridSEQ(const Grid& current, Grid& next)
{
    for (size_t x = 0; x < current.size(); ++x)
    {
        for (size_t y = 0; y < current[x].size(); ++y)
        {
            updateSingleCell(current, next, x, y);
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
    enum class Mode { Sequential, Threads, OpenMP };

    // Runtime parameters and defaults
    Mode mode    = Mode::Threads;
    int  threads = 8;
    int  size    = 5;
    int  width   = 800;
    int  height  = 600;

    { // clang-format off
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
                                    required("OMP" ).set(mode, Mode::OpenMP) ) ).doc("Type of parallelism to use (default: Sequential)")
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

    // cli parameters available to use here...

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

    task_thread_pool::task_thread_pool pool(threads);

    sf::Clock clock;
    sf::Time  elapsed = sf::Time::Zero;
    long int  count   = 0;

    /* Do 'game' loop */
    while (window.isOpen())
    {
        // check for close
        handleWindowEvents(window);

        clock.restart();

        switch (mode)
        {
        case Mode::Sequential:
            updateGridSEQ(*current, *next);
            break;
        case Mode::Threads: {
            const auto numThreads = pool.get_num_threads();
            const auto step       = current->size() / numThreads;
            for (size_t start = 0; start < current->size(); start += step)
            {

                pool.submit_detach(
                    [](size_t start, size_t end, Grid* current, Grid* next) {
                        for (size_t xi = start; xi < end; ++xi)
                        {
                            for (size_t y = 0; y < (*current)[xi].size(); ++y)
                            {
                                updateSingleCell(*current, *next, xi, y);
                            }
                        }
                    },
                    start, start + step, current, next);
            }
            pool.wait_for_tasks();
        }
        break;
        case Mode::OpenMP:
            // updateGridOMP(*current, *next);
            break;
        }

        // record the time taken to calculate the next generation
        elapsed += clock.restart();

        // make everything black
        window.clear();

        // draw live cells as white rectangles
        drawGrid(*current, size, window);

        // draw graphics to screen
        window.display();

        // swap pointers, ready for next iteration.
        std::swap(current, next);

        // every 100 generations, print the elapsed time and reset;
        if (count % 100)
        {
            
            switch (mode)
            {
            case Mode::Sequential:
                std::cout << "100 generation took " << elapsed.asMicroseconds() << " μs with a single thread.\r";
                break;
            case Mode::Threads:
                std::cout << "100 generation took " << elapsed.asMicroseconds() << " μs with " << threads << " std::threads.\r";
                break;
            case Mode::OpenMP:
                std::cout << "100 generation took " << elapsed.asMicroseconds() << " μs with " << threads << " OMP threads.\r";
                break;
            }
            elapsed = sf::Time::Zero;
        }
        count++;
    }

    return EXIT_SUCCESS;
}
