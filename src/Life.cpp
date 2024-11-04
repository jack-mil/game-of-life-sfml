/*
Author: Jackson Miller
Class: ECE6122 A
Last Date Modified: 2024-10-10

Description:
Implementation of Game of Life rules. Does not display the simulated world.
*/

#include <random>

#include "Life.hpp"
#include "Mode.hpp"

/** Only constructor for Life class */
Life::Life(size_t rows, size_t cols, Mode mode, uint threads)
    : m_height(rows),                          // height in cells
      m_width(cols),                           // width in cells
      m_mode{mode},                            // Mode enum specifies the Cuda memory copy technique
      m_threads{threads},                      // save number of std::threads as member
      m_chunkSize{m_height / threads},         // Chunk size for std::thread pooling
      m_bfr_current(rows * cols, State::Dead), // allocate grid buffers with dead cells
      m_bfr_next(rows * cols, State::Dead)     // second buffer
{

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
        cell = static_cast<State>(coin_flip(gen));
    }
}

/**
 * Run one iteration of the Game of Life
 * Use different parallelization techniques according to the current mode
 */
void Life::doOneGeneration()
{
    // Pick an implementation
    switch (m_mode) {
    case Mode::Normal:
        this->updateCudaNormal();
        break;
    case Mode::Managed:
        this->updateCudaManaged();
        break;
    case Mode::Pinned:
        this->updateCudaPinned();
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
            if (this->getCell(row, col) == State::Alive) {
                liveCells.emplace_back(row, col);
            }
        }
    }
    return liveCells;
}

void Life::updateCudaManaged()
{
}
void Life::updateCudaNormal()
{
}
void Life::updateCudaPinned()
{
}
/**
 * Run the game of life rules for the specified rows
 * @param start_row first row to process
 * @param end_row last row to process
 */
inline void Life::process_chunk(size_t start_row, size_t end_row)
{
    for (size_t row = start_row; row < end_row; ++row) {
        for (size_t col = 0; col < this->m_width; ++col) {
            const auto& state = this->simulateSingleCell(row, col);
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
inline Life::State Life::simulateSingleCell(size_t row, size_t col) const
{
    uint neighbors = countNeighbors(row, col);

    if (this->getCell(row, col) == State::Alive) // currently alive
    {
        if (neighbors < 2 || neighbors > 3) {
            return State::Dead; // Cell dies
        }
        else {
            return State::Alive; // Continues to live
        }
    }
    else // currently dead
    {
        if (neighbors == 3) {
            return State::Alive; // Cell becomes alive
        }
        else {
            return State::Dead; // Remains dead
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
    int live_count = 0;

    const long row_s = static_cast<long>(row);
    const long col_s = static_cast<long>(col);

    // Compile-time casts required for Enum -> int conversion
    live_count += static_cast<int>(this->getCellWrap(row_s + 1, col_s - 1)); // top-left
    live_count += static_cast<int>(this->getCellWrap(row_s + 1, col_s));     // top
    live_count += static_cast<int>(this->getCellWrap(row_s + 1, col_s + 1)); // top-right

    live_count += static_cast<int>(this->getCellWrap(row_s, col_s - 1)); // left
    live_count += static_cast<int>(this->getCellWrap(row_s, col_s + 1)); // right

    live_count += static_cast<int>(this->getCellWrap(row_s - 1, col_s - 1)); // bottom-left
    live_count += static_cast<int>(this->getCellWrap(row_s - 1, col_s));     // bottom
    live_count += static_cast<int>(this->getCellWrap(row_s - 1, col_s + 1)); // bottom right

    return live_count;
}

/**
 * Convert row,col specifier to the 1D vector access
 * Reads from current state
 */
inline Life::State Life::getCell(size_t row, size_t col) const
{
    return m_bfr_current[row * m_width + col];
}

/**
 * Convert row,col specifier to the 1D vector access
 * Reads from current state. Allows negative or overflow values to wrap around.
 * This has been optimized specifically for the countNeighbors operation,
 * after checking performance counting.
 */
inline Life::State Life::getCellWrap(long row, long col) const
{
    // row = (row + m_height) % m_height;
    // col = (col + m_width) % m_width;
    // branches were faster than modulo arithmetic
    // guessing branch prediction does wonders here
    if (row < 0) [[unlikely]] {
        row = m_height;
    };
    if (col < 0) [[unlikely]] {
        col = m_width;
    };
    if (row > static_cast<long>(m_height) - 1) [[unlikely]] {
        row = 0;
    };
    if (col > static_cast<long>(m_width) - 1) [[unlikely]] {
        col = 0;
    };
    return m_bfr_current[row * m_width + col];
}

/**
 * Convert row,col specifier to the 1D vector access.
 * Changes contents of next buffer
 * */
inline void Life::setCell(size_t row, size_t col, State state)
{
    m_bfr_next[row * m_width + col] = state;
}
