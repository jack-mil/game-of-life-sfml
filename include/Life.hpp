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

using Alive = char;

// Type alias to save space
using Grid = std::vector<std::vector<Alive>>;


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
    Life(int rows, int cols, Mode mode = Mode::Sequential);
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

    /** Parallelization technique to use */
    gol::Mode m_mode;

    /** Buffer with current state */
    Grid m_bfr_current;
    /** Next buffer to write to */
    Grid m_bfr_next;
};
