/*
Author: Jackson Miller
Class: ECE6122 A
Last Date Modified: 2024-10-09

Description:
Implementation of Game of Life rules. Does not display anything.
*/

#include <random>

#include <omp.h>

#include "Life.hpp"

// forward declare some implementation details
// that don't need to be class members or
// exposed in header
void updateGridOMP(const Grid&, Grid&, int);
void updateGridSEQ(const Grid&, Grid&);

inline int  countNeighbors(const Grid&, size_t, size_t);
inline Alive simulateSingleCell(const Grid&, size_t, size_t);

Life::Life(int rows, int cols, Mode mode)
    : m_mode{mode}
{
    // allocate vectors
    m_bfr_current = Grid(rows, std::vector<Alive>(cols, 0));
    m_bfr_next    = Grid(rows, std::vector<Alive>(cols, 0));

    // Start with a random initial state
    this->seedRandom();
}

/**
 * Initialize each cell of the grid with a random true/false value
 * Makes use of the bernoulli distribution and hardware random number generator.
 * */
void Life::seedRandom()
{
    std::default_random_engine  gen(std::random_device{}());
    std::bernoulli_distribution coin_flip(0.5); // uniform boolean true/false distribution
    for (auto& row : m_bfr_current) {
        for (auto& cell : row) {
            cell = static_cast<Alive>(coin_flip(gen));
        }
    }
}

/**
 * Run one iteration of the Game of Life
 * Use different parallelization techniques according to the current mode
 */
void Life::updateLife()
{
    // Pick an implementation
    switch (m_mode) {
    case Mode::Sequential:
        updateGridSEQ(m_bfr_current, m_bfr_next);
        break;
    case Mode::Threads:
        break;
    case Mode::OpenMP:
        break;
    }

    // swap the std::vectors. This only swaps the underlying pointers,
    // not the contained data. Very cheap and fast (hopefully)
    std::swap(m_bfr_current, m_bfr_next);
}

/**
 * Return a collection of all the row, col points that are currently alive.
 * Compiler will use RVO and move semantics so this avoids unnecessary copying (probably)
 */
std::vector<std::pair<int, int>> Life::getLiveCells() const
{
    std::vector<std::pair<int, int>> liveCells;
    for (size_t row = 0; row < m_bfr_current.size(); ++row) {
        for (size_t col = 0; col < m_bfr_current.at(row).size(); ++col) {
            if (m_bfr_current.at(row).at(col)) {
                liveCells.emplace_back(row, col);
            }
        }
    }
    return liveCells;
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
    for (size_t x = 0; x < current.size(); ++x) {
        for (size_t y = 0; y < current[x].size(); ++y) {
            next[x][y] = simulateSingleCell(current, x, y);
        }
    }
}

/**
 * Given the current state, write the next state to `next`,
 * according to the classic Game of Life rules.
 * Uses OpenMP parallelism
 * @param current
 * @param next
 */
void updateGridOMP(const Grid& current, Grid& next, int numThreads)
{
    omp_set_num_threads(numThreads);

#pragma omp parallel for schedule(static)
    for (size_t x = 0; x < current.size(); ++x) {
        for (size_t y = 0; y < current[x].size(); ++y) {
            next[x][y] = simulateSingleCell(current, x, y);
        }
    }
}

/**
 * Applies the Game of Life rules on a single cell, and return the next state
 *
 * @param current Read only access the the current state (for counting neighbors)
 * @param row pos of the cell
 * @param col pos of the cell
 */
inline Alive simulateSingleCell(const Grid& current, size_t row, size_t col)
{
    uint neighbors = countNeighbors(current, row, col);

    if (current[row][col]) // currently alive
    {
        if (neighbors < 2 || neighbors > 3) {
            return 0; // Cell dies
        }
        else {
            return 1; // Continues to live
        }
    }
    else // currently dead
    {
        if (neighbors == 3) {
            return 1; // Cell becomes alive
        }
        else {
            return 0; // Remains dead
        }
    }
}

/**
 * Find the number of live cells around a point x,y on the grid
 *
 * @param grid world to check
 * @param row pos of the cell
 * @param col pos of the cell
 * @return uint number of neighbors
 */
inline int countNeighbors(const Grid& grid, size_t row, size_t col)
{
    int count = 0;
    // This probably gets unrolled
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            // skip checking the center cell
            if (i || j) [[likely]] {
                Grid::size_type ny = (row + i + grid.size()) % grid.size();
                Grid::size_type nx = (col + j + grid[row].size()) % grid[row].size();

                count += static_cast<int>(grid[ny][nx]); // interpret bool as 1/0
            }
        }
    }
    return count;
}