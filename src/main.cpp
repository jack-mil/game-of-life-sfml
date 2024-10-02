#include <SFML/Graphics.hpp>
#include <cstdlib>
#include <ctime>
#include <random>
#include <vector>

constexpr int WINDOW_WIDTH  = 800;
constexpr int WINDOW_HEIGHT = 600;
constexpr int PIXEL_SIZE    = 5;
constexpr int GRID_WIDTH    = WINDOW_WIDTH / PIXEL_SIZE;
constexpr int GRID_HEIGHT   = WINDOW_HEIGHT / PIXEL_SIZE;

using Grid = std::vector<std::vector<bool>>;

/**
 * @brief Initialize each cell of the grid with a random true/false value
 * Makes use of the bernoulli distribution and hardware random number generator.
 * 
 * @param grid reference to 2D Vector of bools, modified in place.
 */
void seedRandomGrid(Grid& grid)
{
    std::default_random_engine  gen(std::random_device{}());
    std::bernoulli_distribution coin_flip(0.5);
    for (auto& row : grid)
    {
        for (auto cell : row)
        {
            cell = coin_flip(gen);
        }
    }
}

int countNeighbors(const Grid& grid, Grid::size_type x, Grid::size_type y)
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
            Grid::size_type nx = (x + i + GRID_WIDTH) % GRID_WIDTH;
            Grid::size_type ny = (y + j + GRID_HEIGHT) % GRID_HEIGHT;
            count += static_cast<uint>(grid[nx][ny]);
        }
    }
    return count;
}

void updateGrid(Grid& grid, Grid& newGrid)
{
    for (int x = 0; x < GRID_WIDTH; ++x)
    {
        for (int y = 0; y < GRID_HEIGHT; ++y)
        {
            int neighbors = countNeighbors(grid, x, y);

            if (grid[x][y])
            {
                if (neighbors < 2 || neighbors > 3)
                {
                    newGrid[x][y] = false; // Cell dies
                }
            }
            else
            {
                if (neighbors == 3)
                {
                    newGrid[x][y] = true; // Cell becomes alive
                }
            }
        }
    }
}

int main()
{
    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Game of Life");
    window.setFramerateLimit(120); // Set frame rate to control speed

    Grid grid_current(GRID_WIDTH, std::vector<bool>(GRID_HEIGHT, false));
    Grid grid_next(GRID_WIDTH, std::vector<bool>(GRID_HEIGHT, false));

    Grid &refCurrent = grid_current, &refNext = grid_next;

    seedRandomGrid(grid_current);

    unsigned long count = 0;
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                window.close();
            }
            if (event.type == sf::Event::KeyPressed)
            {
                if (event.key.code == sf::Keyboard::Escape)
                {
                    window.close();
                }
            }
        }

        if (count++ % 2)
        {
            refCurrent = grid_next;
            refNext    = grid_current;
        }
        else
        {
            refCurrent = grid_current;
            refNext    = grid_next;
        }

        updateGrid(grid_current, refNext);

        window.clear();

        for (int x = 0; x < GRID_WIDTH; ++x)
        {
            for (int y = 0; y < GRID_HEIGHT; ++y)
            {
                if (refNext[x][y])
                {
                    sf::RectangleShape cell(sf::Vector2f(PIXEL_SIZE, PIXEL_SIZE));
                    cell.setPosition(x * PIXEL_SIZE, y * PIXEL_SIZE);
                    cell.setFillColor(sf::Color::White);
                    window.draw(cell);
                }
            }
        }

        window.display();
    }

    return 0;
}
