/*
Author: Jackson Miller
Class: ECE6122 A
Last Date Modified: 2024-10-10

Description:
Implementation of Game of Life rules. Does not display the simulated world.
*/

#include <random>

#include <omp.h>

// Simple thread pooling implementation
// https://github.com/alugowski/task-thread-pool
#include <task-thread-pool.hpp>

#include "Life.hpp"

/** Only constructor for Life class */
Life::Life(size_t rows, size_t cols, Mode mode, uint threads)
    : m_height(rows),                          // height in cells
      m_width(cols),                           // width in cells
      m_mode{mode},                            // Mode enum specifies the multi-threading technique
      m_threads{threads},                      // save number of std::threads as member
      m_chunkSize{m_height / threads},         // Chunk size for std::thread pooling
      m_bfr_current(rows * cols, State::Dead), // allocate grid buffers with dead cells
      m_bfr_next(rows * cols, State::Dead)     // second buffer
{

    // Start with a random initial state
    this->seedRandom();

    // create thread pool if using that mode
    if (m_mode == Mode::Threads) {
        m_pool_ptr = new task_thread_pool::task_thread_pool{threads};
    }
}

/** Destructor cleans up thread pool (if used) */
Life::~Life()
{
    // Call destructor to stop threads, and cleanup pool object
    delete m_pool_ptr;
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
    case Mode::Sequential:
        this->updateGridSEQ();
        break;
    case Mode::OpenMP:
        this->updateGridOMP();
        break;
    case Mode::Threads:
        this->updateGridThreads();
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

/**
 * Calculate one tick of evolution, according to the classic Game of Life rules
 * Writes the new state to an internal buffer
 *
 * Does not use any multithreading
 */
void Life::updateGridSEQ()
{
    process_chunk(0, m_height);
}

/**
 * Calculate one tick of evolution, according to the classic Game of Life rules
 * Writes the new state to an internal buffer
 *
 * Uses OpenMP parallelism
 */
void Life::updateGridOMP()
{
    // This has to be called just before a omp parallel region
    // I don't know how OMP handles this function being called in a loop...
    // Does it pool the threads? Hopefully. I had to duplicate the for loop code as well...
    omp_set_num_threads(m_threads);
#pragma omp parallel for schedule(static)
    for (size_t row = 0; row < m_height; ++row) {
        for (size_t col = 0; col < m_width; ++col) {
            const auto& state = this->simulateSingleCell(row, col);
            this->setCell(row, col, state);
        }
    }
}

/**
 * Calculate one tick of evolution, according to the classic Game of Life rules
 * Writes the new state to an internal buffer
 *
 * Uses a pool of std::threads
 */
void Life::updateGridThreads()
{
    // split the rows into several chunks, and dispatch threads to handle each section
    for (size_t start = 0; start < m_height; start += m_chunkSize) {

        m_pool_ptr->submit_detach(&Life::process_chunk, this, start, start + m_chunkSize);
    }

    m_pool_ptr->wait_for_tasks();
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

/**
 * Convert row,col specifier to the 1D vector access
 * Reads from current state
 */
inline Life::State Life::getCell(size_t row, size_t col) const
{
    return m_bfr_current.at(row * m_width + col);
}

/**
 * Convert row,col specifier to the 1D vector access.
 * Changes contents of next buffer
 * */
inline void Life::setCell(size_t row, size_t col, State state)
{
    m_bfr_next.at(row * m_width + col) = state;
}
