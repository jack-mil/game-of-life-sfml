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
    App(size_t width, size_t height, size_t cellSize,
        Mode mode    = Mode::Sequential,
        uint threads = 8u,
        bool no_gui  = false);

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

    /** Game of Life simulator */
    Life m_life;

    /** Drawable to represent a living cell */
    sf::RectangleShape m_cellSprite;

    /** mode this application is running in */
    const Mode m_mode;

    /** Number of threads to use in OMP or THRD mode */
    const uint m_threads;

    /** Whether a window should be displayed, or just print the processing timings */
    const bool m_no_gui;
};
