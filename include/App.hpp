#pragma once

#include <SFML/Graphics.hpp>

#include "Life.hpp"
#include "Mode.hpp"

class App {
  public:
    /**
     * Construct an App to run Game of Life
     *
     * @param width width of the application window (px)
     * @param height height of the application window (px)
     * @param mode what type of parallelization to use
     */
    App(size_t width, size_t height, size_t cellSize, gol::Mode mode, int threadCount = 8);

    /** Entrypoint to game loop. Call this to start the application */
    void run();

  private:
    /** Create the application window and position it*/
    void setupWindow(size_t width, size_t height);

    /** Handle closing the SFML window */
    void handleEvents();

    /** Display the time for simulating 100 iterations */
    void printTimings(sf::Time elapsed);

    /** Represent the Game of Life using SFML graphics */
    void drawLife();

    /** SFML OS window */
    sf::RenderWindow m_window;

    /** mode this application is running in */
    gol::Mode m_mode;

    /** Game of Life simulator */
    Life m_life;

    /** Number of threads to use in OMP or THRD mode */
    int m_threads;

    /** Drawable to represent a living cell */
    sf::RectangleShape m_cellSprite;
};