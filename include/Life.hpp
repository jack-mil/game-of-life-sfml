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
     * @param mode parallelism to use (default none)
     * @param threads number of threads for multithreaded modes (default 8)
     */
    Life(size_t rows, size_t cols, Mode mode = Mode::Sequential, uint threads = 8);

    /** Destructor cleans up thead pool (if used) */
    ~Life();

    /** Disabling moving and copy to follow the 'Rule of Five', and because I don't need it
     * see §C21 of: https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
     */
    Life(const Life&)                = delete; // no copy constructor
    Life(Life&&) noexcept            = delete; // no move constructor
    Life& operator=(const Life&)     = delete; // no copy assignment
    Life& operator=(Life&&) noexcept = delete; // no move assignment

    /** Run a single iteration of the Rules on the current state */
    void doOneGeneration();

    /**
     * Get a vector with position of every currently living cell.
     * Use this to display Game of Life world using any method
     */
    std::vector<std::pair<int, int>> getLiveCells() const;

  private:
    /** 
     * Internal representation of a cell.
     * Can be safely cast to numeric type for counting living cells.
     */
    enum class State : char { Alive = 1, Dead  = 0 };

    /** Type alias to save space */
    using Grid = std::vector<State>;

    /** Setup initial random state */
    void seedRandom();

    /** Get the state of a current cell */
    State getCell(size_t row, size_t col) const;

    /** Optimized method for wrapping around edges */
    State getCellWrap(long row, long col) const;

    /** Set the state for next iteration */
    void setCell(size_t row, size_t col, State state);

    /** Run Game of Life with no multithreading */
    void updateGridSEQ();

    /** Run Game of Life using OpenMP threading */
    void updateGridOMP();

    /** Run Game of Life using std::thread pooling */
    void updateGridThreads();

    /** Actual looping through the grid, used by above methods */
    void process_chunk(size_t start_row, size_t end_row);

    /** Return what the next state of the cell at (row,col) should be */
    State simulateSingleCell(size_t row, size_t col) const;

    /** Return how many living neighbors around a cell at (row,col) */
    int countNeighbors(size_t row, size_t col) const;

    /** Number of cell rows */
    const size_t m_height;

    /** Number of cell columns */
    const size_t m_width;

    /** Parallelization technique to use */
    const Mode m_mode;

    /** Number of threads to use for (both) multithreading modes */
    const uint m_threads;

    /** Number of rows to process per thread in std::thread pooling mode */
    const size_t m_chunkSize;

    /** std::thread pool. Created only if using std::thread pooling mode */
    task_thread_pool::task_thread_pool* m_pool_ptr = nullptr;

    /** Buffer with current state */
    Grid m_bfr_current;

    /** Next buffer to write to */
    Grid m_bfr_next;
};
