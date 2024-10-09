/*
Author: Jackson Miller
Class: ECE6122 A
Last Date Modified: 2024-10-09

Description:
An implementation of the Game of Life rules.
Tries to be display-method agnostic, something else
should display or print the Life world
*/
#pragma once

#include <utility> // std::pair
#include <vector>  // std::vector

#include "Mode.hpp"

using Mode = gol::Mode;

using Alive = unsigned char;

// Type alias to save space
using Grid = std::vector<Alive>;

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
     */
    Life(size_t rows, size_t cols, Mode mode = Mode::Sequential);
    Life() = delete; // no default constructor

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
    void updateGridOMP(int threads);
    /** Run Game of Life using std::thread pooling */

    /** Return what the next state of the cell at (row,col) should be */
    Alive simulateSingleCell(size_t row, size_t col) const;
    /** Return how many living neighbors around a cell at (row,col) */
    int countNeighbors(size_t row, size_t col) const;

    /** Number of cell columns */
    size_t m_width;

    /** Number of cell rows */
    size_t m_height;

    /** Parallelization technique to use */
    gol::Mode m_mode;

    /** Buffer with current state */
    Grid m_bfr_current;
    /** Next buffer to write to */
    Grid m_bfr_next;
};
