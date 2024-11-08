/*
Author: Jackson Miller
Class: ECE6122 A
Last Date Modified: 2024-10-10

Description:
Control the SFML OS window and display the Game of Life simulation on the screen
*/
#include <iostream>

#include <SFML/Graphics.hpp>

#include "Life.cuh" // Game of Life simulator
#include "Mode.hpp" // Operating mode enum

#include "App.hpp"

App::App(size_t width, size_t height, size_t cellSize, Mode mode, uint threads, bool no_gui)
    : m_life{width / cellSize, height / cellSize, mode, threads},
      m_cellSprite{sf::Vector2f(cellSize, cellSize)},
      m_mode{mode},
      m_threads{threads},
      m_no_gui{no_gui}
{   
    // Only do SFML stuff if running in "GUI" mode
    if (!no_gui) {
        setupWindow(width, height);
    }

    // imbue locale (shows numbers with commas, as configured by user's OS)
    std::cout.imbue(std::locale(""));
}

/** 
 * Create and position the OS window with a given width and height (in pixels)
 * 
 * @param width horizontal size
 * @param height vertical size
 */
void App::setupWindow(size_t width, size_t height)
{
    m_window.create(sf::VideoMode(width, height), "Game of Life CUDA - Jackson Miller - ECE6122");

    // set some OS window options
    m_window.setMouseCursorVisible(false);
    m_window.setVerticalSyncEnabled(false);
    m_window.setFramerateLimit(60);

    // place the window in the center of the desktop
    const auto& desktop = sf::VideoMode::getDesktopMode();
    const auto  xpos    = (desktop.width / 2u) - (m_window.getSize().x / 2u);
    const auto  ypos    = (desktop.height / 2u) - (m_window.getSize().y / 2u);
    m_window.setPosition(sf::Vector2i(static_cast<int>(xpos), static_cast<int>(ypos)));
}

/** Application SFML game loop (loop while window is open) */
void App::run()
{
    sf::Clock timer;
    sf::Time  elapsed = sf::Time::Zero;

    long int iterations = 0;

    while (m_window.isOpen() || m_no_gui) { 

        // Simulate the next generation,
        // and time how long it takes with the current mode

        timer.restart();            // timer start
        m_life.doOneGeneration();   // generate Life
        elapsed += timer.restart(); // timer end

        // every 100 iterations, print timing and reset time measurement
        iterations++;
        if (iterations % 100 == 0) {
            printTimings(elapsed);
            elapsed = sf::Time::Zero;
        }

        // quite after 600 iterations when running without a GUI
        if (m_no_gui && iterations > 600) {
            break;
        }

        // Don't do SFML display stuff if no GUI mode
        if (m_no_gui) {
            continue;
        }

        handleEvents();

        // Clear the window (old frame)
        m_window.clear();

        // Represent the current state suing SFML graphics
        drawLife();

        // Display the new frame
        m_window.display();
    }
    
    // Window is closed, time to quit
}

/** Display timing information for each mode */
void App::printTimings(sf::Time elapsed)
{
    switch (m_mode) { // clang-format off
    case Mode::Normal:
        std::cout << "100 generation took " 
                  << elapsed.asMicroseconds() 
                  << " μs with " << m_threads << " threads per block" 
                  << " using Normal memory allocation.\n";
        break;
    case Mode::Managed:
        std::cout << "100 generation took " 
                  << elapsed.asMicroseconds() 
                  << " μs with " << m_threads << " threads per block" 
                  << " using Managed memory allocation.\n";
        break;
    case Mode::Pinned:
        std::cout << "100 generation took " 
                  << elapsed.asMicroseconds() 
                  << " μs with " << m_threads << " threads per block"
                  << " using Pinned memory allocation.\n";
        break;
    } // clang-format on

    elapsed = sf::Time::Zero; // Reset timer
}

/** Process events to close the window or reset the game */
void App::handleEvents()
{
    static sf::Event event;
    while (m_window.pollEvent(event)) {
        if (event.type == sf::Event::Closed) { // close with 'X' button
            m_window.close();
        }
        else if (event.type == sf::Event::KeyPressed) { // close with ESC or ENTER
            if (event.key.code == sf::Keyboard::Escape || event.key.code == sf::Keyboard::Return) {
                m_window.close();
            }
        }
    }
}

/** Draw every living cell as a white SFML rectangle sprite */
void App::drawLife()
{
    static const auto& size = m_cellSprite.getSize();

    // Draw only the living cells at (row, col) (pair<int,int>)
    for (const auto& [row, col] : m_life.getLiveCells()) {
        // Draw the same sprite object at many positions
        m_cellSprite.setPosition(row * size.x, col * size.y);
        m_window.draw(m_cellSprite);
    }
}
