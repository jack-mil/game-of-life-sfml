/*
Author: Jackson Miller
Class: ECE6122 A
Last Date Modified: 2024-10-10

Description:
An implementation of the Game of Life rules.
Tries to be display-method agnostic, something else
should display or print the Life world
*/

#pragma once

#include <utility> // std::pair
#include <vector>  // std::vector

#include "Mode.hpp"

using Alive = unsigned char;

// Type alias to save space
using Grid = std::vector<Alive>;

// forward declare thead pooling type
namespace task_thread_pool {
class task_thread_pool;
};

class Life {
  public:
    /** Create a new Game of Life simulation with a random starting state.
     *
     * The universe wraps edges in a toroidal shape.
     * Specify mode to enable multithreading techniques.
     *
     * @param rows number of rows in the universe
     * @param cols number of cols in the universe
     * @param threads number of threads for multithreaded modes (default 8)
     * @param mode parallelism to use (default none)
     */
    Life(size_t rows, size_t cols, Mode mode = Mode::Sequential, uint threads = 8);
    Life() = delete; // no default constructor
    ~Life();

    /** Run a single iteration of the Rules on the current state */
    void updateLife();

    /**
     * Get a vector with position of every currently living cell.
     * Use this to display Game of Life world using any method
     */
    std::vector<std::pair<int, int>> getLiveCells() const;

  private:
    /** Setup initial random state */
    void seedRandom();

    /** Get the state of a current cell */
    Alive getCell(size_t row, size_t col) const;
    /** Set the state for next iteration */
    void setCell(size_t row, size_t col, Alive state);

    /** Run Game of Life with no multithreading */
    void updateGridSEQ();
    /** Run Game of Life using OpenMP threading */
    void updateGridOMP();
    /** Run Game of Life using std::thread pooling */
    void updateGridThreads();
    void process_chunk(size_t start_row, size_t end_row);

    /** Return what the next state of the cell at (row,col) should be */
    Alive simulateSingleCell(size_t row, size_t col) const;
    /** Return how many living neighbors around a cell at (row,col) */
    int countNeighbors(size_t row, size_t col) const;

    /** Number of cell columns */
    const size_t m_width;

    /** Number of cell rows */
    const size_t m_height;

    /** Parallelization technique to use */
    const Mode m_mode;

    /** Number of threads to use for multithreading modes */
    const uint m_threads;

    const size_t m_chunkSize;

    /** Thread pool setup only if using std::thread pooling mode */
    task_thread_pool::task_thread_pool* m_pool_ptr = nullptr;

    /** Buffer with current state */
    Grid m_bfr_current;
    /** Next buffer to write to */
    Grid m_bfr_next;
};
