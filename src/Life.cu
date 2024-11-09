/*
Author: Jackson Miller
Class: ECE6122 A
Last Date Modified: 2024-10-10

Description:
Implementation of Game of Life rules. Does not display the simulated world.
*/

#include <random>

#include <cuda_runtime.h>

#include "Life.cuh"
#include "Mode.hpp"
#include "helper_cuda.cuh"
#include <cassert>

size_t d_current_pitch; // width in bytes of allocation
size_t d_next_pitch;    // width in bytes of allocation

/** Only constructor for Life class */
Life::Life(size_t height, size_t width, Mode mode, uint threads)
    : m_height{height},       // height in cells
      m_width{width},         // width in cells
      m_mode{mode},           // Mode enum specifies the Cuda memory copy technique
      m_threads{nullptr},     // Number of threads per execution block
      m_blocks{nullptr},      // Number of blocks in the grid (round up)
      m_bfr_current{nullptr}, // allocate grid buffers with dead cells
      m_bfr_next{nullptr}     // second buffer
//   m_bfr_current(rows * cols, State::Dead),           // allocate grid buffers with dead cells
//   m_bfr_next(rows * cols, State::Dead)               // second buffer
{

    int devID = findCudaDevice();

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    if (!deviceProp.managedMemory) {
        // Game of life requires being run on a device that supports Unified Memory
        fprintf(stderr, "Unified Memory not supported on this device\n");
        exit(EXIT_WAIVED);
    }

    // Statistics about the GPU device
    printf(
        "> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
        deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    assert(mode == Mode::Normal);

    m_bfr_current = new State[m_width * m_height];
    m_bfr_next    = new State[m_width * m_height];

    // Allocte device memory (host already allocated as a vector<State>)
    // checkCudaErrors(cudaMalloc(&d_bfr_current, sizeof(State) * m_width * m_height));
    // checkCudaErrors(cudaMalloc(&d_bfr_next, sizeof(State) * m_height * m_width));

    checkCudaErrors(cudaMallocPitch(&d_bfr_current, &d_current_pitch,
                                    sizeof(State) * m_width, m_height));
    checkCudaErrors(cudaMallocPitch(&d_bfr_next, &d_next_pitch,
                                    sizeof(State) * m_width, m_height));

    // calculate 2D thread and block size
    // we want maximum parallelism,
    // so try to get a thread for every cell (x,y)
    m_threads = new dim3{threads, threads}; // AKA block size
    // need enough blocks for a thread for every cell
    m_blocks = new dim3{
        ((uint)m_width + m_threads->x - 1) / m_threads->x,
        ((uint)m_height + m_threads->y - 1) / m_threads->y,
    }; // AKA Grid size

    // Seed the host starting universe with a random state
    this->seedRandom();
}

/** Destructor frees allocated memory */
Life::~Life()
{
    // free heap data
    delete m_threads;
    delete m_blocks;
    delete m_bfr_current;
    delete d_bfr_next;
    // free Cuda device memory
    checkCudaErrors(cudaFree(d_bfr_current));
    checkCudaErrors(cudaFree(d_bfr_next));
}

/**
 * Initialize each cell of the grid with a random true/false value
 * Makes use of the bernoulli distribution and hardware random number generator.
 * */
void Life::seedRandom()
{
    std::default_random_engine  gen(std::random_device{}());
    std::bernoulli_distribution coin_flip(0.5); // uniform boolean true/false distribution
    for (size_t i = 0; i < m_width * m_height; i++) {
        m_bfr_current[i] = static_cast<State>(coin_flip(gen));
    }
    // for (auto& cell : m_bfr_current) {
    //     cell = static_cast<State>(coin_flip(gen));
    // }
}

/**
 * Run one iteration of the Game of Life
 * Use different parallelization techniques according to the current mode
 */
void Life::doOneGeneration()
{
    // Pick an implementation
    switch (m_mode) {
    case Mode::Normal:
        this->updateCudaNormal();
        break;
    case Mode::Managed:
        this->updateCudaManaged();
        break;
    case Mode::Pinned:
        this->updateCudaPinned();
        break;
    }

    // swap the std::vectors. This only swaps the underlying pointers,
    // not the contained data. Very cheap and fast (hopefully)
    // std::swap(m_bfr_current, m_bfr_next);
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

__global__ void deviceOneGeneration(uint8_t* now, uint8_t* next,
                                    size_t pitch_now, size_t pitch_next,
                                    int width, int height)
{
    // "2d" arrangement of threads and blocks
    unsigned int x_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int y_index = (blockIdx.y * blockDim.y) + threadIdx.y;

    // printf("px=%ud,%ud\n", x_index, y_index);
    // printf("Tidx=%d, Tidy=%d, Tidz=%d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    // printf("blkX=%d, blkY=%d, blkZ=%d\n", blockDim.x, blockDim.y, blockDim.z);
    // printf("gridX=%d, gridY=%d, gridZ=%d\n", gridDim.x, gridDim.y, gridDim.z);

    /* total number of spawned threads in x direction*/
    int stride_col = blockDim.x * gridDim.x;
    /* total number of spawned threads in y direction*/
    int stride_row = blockDim.y * gridDim.y;

    // if (x_index == 0 || x_index >= width) {
    //     return;
    // }
    // if (y_index == 0 || y_index >= height) {
    //     return;
    // }

    // 2d stride loop. Only executed more than once if not enough threads
    for (size_t y = y_index; y < height; y += stride_row) {
        for (size_t x = x_index; x < width; x += stride_col) {
            // T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
            uint8_t* top_row = now + ((y - 1) * pitch_now);
            uint8_t* mid_row = now + (y * pitch_now);
            uint8_t* bot_row = now + ((y + 1) * pitch_now);

            // sum of whole 9x9 area
            int count = top_row[x - 1] + top_row[x] + top_row[x + 1] +
                        mid_row[x - 1] + mid_row[x] + mid_row[x + 1] +
                        bot_row[x - 1] + bot_row[x] + bot_row[x + 1];

            // uint8_t* next_cell = (uint8_t*)next + (y * pitch_next) + x;

            next[(y * pitch_next) + x] = (count == 3)   ? 1u         // alive
                                         : (count == 4) ? mid_row[x] // no change
                                                        : 0u;        // dies
            // next[(y * pitch_next) + x] = (count == 3 || (count == 2 && mid_row[x])) ? 1u : 0u;
        }
    }
    // size_t ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
    // size_t iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
    // size_t id = iy * (width + 2) + ix;
}

void Life::updateCudaNormal()
{
    // copy host array (vector) to device
    checkCudaErrors(cudaMemcpy(d_bfr_current, m_bfr_current,
                               sizeof(State) * m_width * m_height,
                               cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(d_bfr_current, m_bfr_current.data(),
    //                            sizeof(State) * m_bfr_current.size(),
    //                            cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy2D(d_bfr_current, d_current_pitch,
    //                              m_bfr_current.data(), sizeof(State) * m_width,
    //                              sizeof(State) * m_width, m_height,
    //                              cudaMemcpyHostToDevice));
    // execute kernel
    deviceOneGeneration<<<*m_blocks, *m_threads>>>((uint8_t*)d_bfr_current, (uint8_t*)d_bfr_next,
                                                   d_current_pitch, d_next_pitch,
                                                   m_width, m_height);

    // copy memory back to device (blocks until kernel done)
    checkCudaErrors(cudaMemcpy(m_bfr_next, d_bfr_next,
                               sizeof(State) * m_height * m_width,
                               cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(m_bfr_next.data(), d_bfr_next,
    //                            sizeof(State) * m_bfr_next.size(),
    //                            cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy2D(m_bfr_next.data(), sizeof(State) * m_width, /* pitch==width on host */
    //                              d_bfr_next, d_next_pitch,
    //                              sizeof(State) * m_width, m_height,
    //                              cudaMemcpyDeviceToHost));
    // new gen stored in bfr_next
    // swap the std::vectors. This only swaps the underlying pointers,
    // not the contained data. Very cheap and fast (hopefully)
    std::swap(m_bfr_current, m_bfr_next);
}
void Life::updateCudaManaged()
{
}
void Life::updateCudaPinned()
{
}

/**
 * Run the game of life rules for the specified rows
 * @param start_row first row to process
 * @param end_row last row to process
 */
// inline void Life::process_chunk(size_t start_row, size_t end_row)
// {
//     for (size_t row = start_row; row < end_row; ++row) {
//         for (size_t col = 0; col < this->m_width; ++col) {
//             const auto& state = this->simulateSingleCell(row, col);
//             this->setCell(row, col, state);
//         }
//     }
// }

/**
 * Applies the Game of Life rules on a single cell, and return the next state
 *
 * @param current Read only access the the current state (for counting neighbors)
 * @param row pos of the cell
 * @param col pos of the cell
 */
// inline Life::State Life::simulateSingleCell(size_t row, size_t col) const
// {
//     uint neighbors = countNeighbors(row, col);

//     if (this->getCell(row, col) == State::Alive) // currently alive
//     {
//         if (neighbors < 2 || neighbors > 3) {
//             return State::Dead; // Cell dies
//         }
//         else {
//             return State::Alive; // Continues to live
//         }
//     }
//     else // currently dead
//     {
//         if (neighbors == 3) {
//             return State::Alive; // Cell becomes alive
//         }
//         else {
//             return State::Dead; // Remains dead
//         }
//     }
// }

/**
 * Find the number of live cells around a point x,y on the grid
 *
 * @param grid world to check
 * @param row pos of the cell
 * @param col pos of the cell
 * @return number of neighbors
 */
// inline int Life::countNeighbors(size_t row, size_t col) const
// {
//     int live_count = 0;

//     const long row_s = static_cast<long>(row);
//     const long col_s = static_cast<long>(col);

//     // Compile-time casts required for Enum -> int conversion
//     live_count += static_cast<int>(this->getCellWrap(row_s + 1, col_s - 1)); // top-left
//     live_count += static_cast<int>(this->getCellWrap(row_s + 1, col_s));     // top
//     live_count += static_cast<int>(this->getCellWrap(row_s + 1, col_s + 1)); // top-right

//     live_count += static_cast<int>(this->getCellWrap(row_s, col_s - 1)); // left
//     live_count += static_cast<int>(this->getCellWrap(row_s, col_s + 1)); // right

//     live_count += static_cast<int>(this->getCellWrap(row_s - 1, col_s - 1)); // bottom-left
//     live_count += static_cast<int>(this->getCellWrap(row_s - 1, col_s));     // bottom
//     live_count += static_cast<int>(this->getCellWrap(row_s - 1, col_s + 1)); // bottom right

//     return live_count;
// }

/**
 * Convert row,col specifier to the 1D vector access
 * Reads from current state
 */
inline Life::State Life::getCell(size_t row, size_t col) const
{
    return m_bfr_current[row * m_width + col];
}

// /**
//  * Convert row,col specifier to the 1D vector access
//  * Reads from current state. Allows negative or overflow values to wrap around.
//  * This has been optimized specifically for the countNeighbors operation,
//  * after checking performance counting.
//  */
// inline Life::State Life::getCellWrap(long row, long col) const
// {
//     // row = (row + m_height) % m_height;
//     // col = (col + m_width) % m_width;
//     // branches were faster than modulo arithmetic
//     // guessing branch prediction does wonders here
//     if (row < 0) [[unlikely]] {
//         row = m_height;
//     };
//     if (col < 0) [[unlikely]] {
//         col = m_width;
//     };
//     if (row > static_cast<long>(m_height) - 1) [[unlikely]] {
//         row = 0;
//     };
//     if (col > static_cast<long>(m_width) - 1) [[unlikely]] {
//         col = 0;
//     };
//     return m_bfr_current[row * m_width + col];
// }

// /**
//  * Convert row,col specifier to the 1D vector access.
//  * Changes contents of next buffer
//  * */
// inline void Life::setCell(size_t row, size_t col, State state)
// {
//     m_bfr_next[row * m_width + col] = state;
// }
