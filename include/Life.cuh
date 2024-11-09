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

/** forward declaration to avoid including cuda_runtime in this file. */
struct dim3;

class Life {
  public:
    /** Create a new Game of Life simulation with a random starting state.
     *
     * The universe wraps edges in a toroidal shape.
     * The Life class uses CUDA processing in various modes
     *
     * @param rows number of rows in the universe
     * @param cols number of cols in the universe
     * @param mode CUDA memory copy mode to use (default Normal)
     * @param threads number of CUDA threads per block (default 32)
     */
    Life(size_t rows, size_t cols, Mode mode = Mode::Normal, uint threads = 32);
    Life() = delete;

    /** Destructor cleans up thead pool (if used) */
    ~Life();

    /** Disabling moving and copy to follow the 'Rule of Five', and because I don't need it
     * see §C21 of: https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
     */
    Life(const Life&)                = delete; // no copy constructor
    Life(Life&&) noexcept            = delete; // no move constructor
    Life& operator=(const Life&)     = delete; // no copy assignment
    Life& operator=(Life&&) noexcept = delete; // no move assignment

    /**
     * Internal representation of a cell.
     * Can be safely cast to numeric type for counting living cells.
     */
    enum class State : uint8_t { Alive = 1, Dead  = 0 };

    /** Run a single iteration of the Rules on the current state */
    void doOneGeneration();

    /**
     * Get a vector with position of every currently living cell.
     * Use this to display Game of Life world using any method
     */
    std::vector<std::pair<int, int>> getLiveCells() const;

  private:
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

    void updateCudaManaged();

    void updateCudaPinned();

    void updateCudaNormal();

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

    /** Number of threads per block to use */
    const size_t m_threads;
    /** Number of blocks per grid to use */
    const size_t m_blocks;

    /** Buffer with current state */
    Grid m_bfr_current;

    /** Next buffer to write to */
    Grid m_bfr_next;

    State* d_bfr_current = nullptr;
    State* d_bfr_next    = nullptr;

};
