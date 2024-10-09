/*
Author: Jackson Miller
Class: ECE6122 A
Last Date Modified: 2024-10-09

Description:
Control the SFML OS window and display the Game of Life simulation on the screen
*/
#include <iostream>

#include <SFML/Graphics.hpp>

#include "App.hpp"
#include "Life.hpp"
#include "Mode.hpp"

using Mode = gol::Mode;

App::App(size_t width, size_t height, size_t cellSize, Mode mode, int threads)
    : m_mode{mode},
      m_life{width / cellSize, height / cellSize, mode, threads},
      m_threads{threads},
      m_cellSprite{sf::Vector2f(cellSize, cellSize)}
{
    setupWindow(width, height);
    m_cellSprite.setFillColor(sf::Color::White);
}

void App::setupWindow(size_t width, size_t height)
{
    m_window.create(sf::VideoMode(width, height), "Game of Life - Jackson Miller - ECE6122");

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

    while (m_window.isOpen()) {
        handleEvents();

        // Simulate the next generation,
        // and time how long it takes with the current mode
        timer.restart();
        m_life.updateLife();
        elapsed += timer.restart();

        // every 100 iterations, print timing and reset time measurement
        if (iterations % 100 == 0) {
            printTimings(elapsed);
            elapsed = sf::Time::Zero;
        }
        iterations++;

        // Clear the window (old frame)
        m_window.clear();

        // Represent the current state suing SFML graphics
        drawLife();

        // Display the new frame
        m_window.display();
    }
}

/** Display timing information for each mode */
void App::printTimings(sf::Time elapsed)
{
    switch (m_mode) {
    case Mode::Sequential:
        std::cout << "100 generation took " << elapsed.asMicroseconds() << " μs with a single thread." << std::endl;
        break;
    case Mode::Threads:
        std::cout << "100 generation took " << elapsed.asMicroseconds() << " μs with " << m_threads << " std::threads." << std::endl;
        break;
    case Mode::OpenMP:
        std::cout << "100 generation took " << elapsed.asMicroseconds() << " μs with " << m_threads << " OMP threads." << std::endl;
        break;
    }
    // Reset timer
    elapsed = sf::Time::Zero;
}

/** Process events to close the window or reset the game */
void App::handleEvents()
{
    static sf::Event event;
    while (m_window.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            m_window.close();
        }
        else if (event.type == sf::Event::KeyPressed) {
            if (event.key.code == sf::Keyboard::Escape || event.key.code == sf::Keyboard::Return) {
                m_window.close();
            }
        }
    }
}

/** Draw every living cell as a white SFML rectangle sprite */
void App::drawLife()
{
    const auto& size = m_cellSprite.getSize();

    // Draw only the living cells at the (row,col) (pair<int,int>)
    for (const auto& [row, col] : m_life.getLiveCells()) {
        // Draw the same sprite object at many positions
        m_cellSprite.setPosition(row * size.x, col * size.y);
        m_window.draw(m_cellSprite);
    }
}