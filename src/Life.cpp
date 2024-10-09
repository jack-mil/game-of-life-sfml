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

Life::Life(size_t rows, size_t cols, Mode mode)
    : m_width(cols),
      m_height(rows),
      m_mode{mode},
      m_bfr_current(rows * cols, 0),
      m_bfr_next(rows * cols, 0)
{
    // allocate vectors
    // m_bfr_current = Grid(rows, std::vector<Alive>(cols, 0));
    // m_bfr_next    = Grid(rows, std::vector<Alive>(cols, 0));

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
    for (auto& cell : m_bfr_current) {
        cell = static_cast<Alive>(coin_flip(gen));
        // for (auto& cell : row) {
        // }
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
        updateGridSEQ();
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
    for (size_t row = 0; row < m_height; ++row) {
        for (size_t col = 0; col < m_width; ++col) {
            if (this->getCell(row, col)) {
                liveCells.emplace_back(row, col);
            }
        }
    }
    return liveCells;
}

/**
 * Convert row,col specifier to the 1D vector access
 * Reads from current state
 */
inline Alive Life::getCell(size_t row, size_t col) const
{
    return m_bfr_current.at(row * m_width + col);
}

/**
 * Convert row,col specifier to the 1D vector access.
 * Changes contents of next buffer
 * */
inline void Life::setCell(size_t row, size_t col, Alive state)
{
    m_bfr_next.at(row * m_width + col) = state;
}

/**
 * Given the current state, write the next state to `next`,
 * according to the classic Game of Life rules.
 * Uses regular sequential processing.
 * @param current
 * @param next
 */
void Life::updateGridSEQ()
{
    for (size_t row = 0; row < m_height; ++row) {
        for (size_t col = 0; col < m_width; ++col) {
            Alive state = simulateSingleCell(row, col);
            this->setCell(row, col, state);
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
void Life::updateGridOMP(int threads)
{
    omp_set_num_threads(threads);

#pragma omp parallel for schedule(static)
    for (size_t row = 0; row < m_height; ++row) {
        for (size_t col = 0; col < m_width; ++col) {
            Alive state = simulateSingleCell(row, col);
            this->setCell(row, col, state);
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
inline Alive Life::simulateSingleCell(size_t row, size_t col) const
{
    uint neighbors = countNeighbors(row, col);

    if (this->getCell(row, col)) // currently alive
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
 * @return number of neighbors
 */
inline int Life::countNeighbors(size_t row, size_t col) const
{
    int count = 0;
    // This probably gets unrolled
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            // skip checking the center cell
            if (i || j) [[likely]] {
                size_t nrow = (row + i + m_height) % m_height;
                size_t ncol = (col + j + m_width) % m_width;

                count += static_cast<int>(this->getCell(nrow, ncol)); // interpret Alive as 1/0
            }
        }
    }
    return count;
}