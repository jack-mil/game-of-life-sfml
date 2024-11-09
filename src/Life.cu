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

/** Only constructor for Life class */
Life::Life(size_t height, size_t width, Mode mode, uint threads)
    : m_height{height},                           // height in cells
      m_width{width},                             // width in cells
      m_mode{mode},                               // Mode enum specifies the Cuda memory copy technique
      m_threads{nullptr},                         // Dim of threads per execution block
      m_blocks{nullptr},                          // Dim of blocks in the grid (round up)
      m_bfr_current(height * width, State::Dead), // allocate grid buffers with dead cells
      m_bfr_next(height * width, State::Dead)     // second buffer
{

    int devID = findCudaDevice();

    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));

    if (!props.managedMemory) {
        // Game of life requires being run on a device that supports Unified Memory
        fprintf(stderr, "Unified Memory not supported on this device\n");
        exit(EXIT_WAIVED);
    }

    if (!threads > props.maxThreadsPerBlock) {
        fprintf(stderr, "Threads per block cannot be greater than: %d", props.maxThreadsPerBlock);
        exit(EXIT_WAIVED);
    }
    printDeviceStats(props);

    assert(mode == Mode::Normal);

    // Allocte device memory (host already allocated as a vector<State>)
    checkCudaErrors(cudaMallocPitch(&d_bfr_current, &d_current_pitch,
                                    sizeof(State) * m_width, m_height));
    checkCudaErrors(cudaMallocPitch(&d_bfr_next, &d_next_pitch,
                                    sizeof(State) * m_width, m_height));

    // assume threads is multiple of 32
    const uint long_side = threads >= 512   ? 32
                           : threads >= 128 ? 16
                                            : 8;

    auto y_dim = threads / long_side;
    auto x_dim = threads / y_dim;
    m_threads  = new dim3{x_dim, y_dim}; // AKA block size
    // need enough blocks for a thread for every cell (round up)
    m_blocks = new dim3{
        ((uint)m_width + m_threads->x - 1) / m_threads->x,
        ((uint)m_height + m_threads->y - 1) / m_threads->y,
    }; // AKA Grid size

    printf("Life %dx%d\n", m_width, m_height);
    printf("CUDA : blockDim=(%u,%u,%u), gridDim=(%u,%u,%u)\n",
           m_threads->x, m_threads->y, m_threads->z,
           m_blocks->x, m_blocks->y, m_blocks->z);

    // Seed the host starting universe with a random state
    this->seedRandom();
}

/** Destructor frees allocated memory */
Life::~Life()
{
    // free heap data
    delete m_threads;
    delete m_blocks;
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
    // for (size_t i = 0; i < m_width * m_height; i++) {
    //     m_bfr_current[i] = static_cast<State>(coin_flip(gen));
    // }
    for (auto& cell : m_bfr_current) {
        cell = static_cast<State>(coin_flip(gen));
    }
}

/**
 * Convert row,col specifier to the 1D vector access
 * Reads from current state
 */
inline Life::State Life::getCell(size_t row, size_t col) const
{
    return m_bfr_current[row * m_width + col];
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
    std::swap(m_bfr_current, m_bfr_next);
}

/* Device kernel using a 2d grid and block size */
__global__ void deviceOneGeneration(uint8_t* now, uint8_t* next,
                                    size_t pitch_now, size_t pitch_next,
                                    int width, int height)
{
    // "2d" arrangement of threads and blocks
    uint x_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint y_index = (blockIdx.y * blockDim.y) + threadIdx.y;

    // printf("px=%ud,%ud\n", x_index, y_index);
    // printf("Tidx=%d, Tidy=%d, Tidz=%d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    // printf("blkX=%d, blkY=%d, blkZ=%d\n", blockDim.x, blockDim.y, blockDim.z);
    // printf("gridX=%d, gridY=%d, gridZ=%d\n", gridDim.x, gridDim.y, gridDim.z);

    /* total number of spawned threads in x direction*/
    uint stride_col = blockDim.x * gridDim.x;
    /* total number of spawned threads in y direction*/
    uint stride_row = blockDim.y * gridDim.y;

    // if (x_index == 0 || x_index >= width) {
    //     return;
    // }
    // if (y_index == 0 || y_index >= height) {
    //     return;
    // }

    // 2d stride loop. Only executed more than once if not enough blocks (unlikely)
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
    checkCudaErrors(cudaMemcpy2D(d_bfr_current, d_current_pitch,
                                 m_bfr_current.data(), sizeof(State) * m_width, // host pitch is width
                                 sizeof(State) * m_width, m_height,
                                 cudaMemcpyHostToDevice));
    // execute kernel
    deviceOneGeneration<<<*m_blocks, *m_threads>>>((uint8_t*)d_bfr_current, (uint8_t*)d_bfr_next,
                                                   d_current_pitch, d_next_pitch,
                                                   m_width, m_height);
    checkCudaErrors(cudaPeekAtLastError()); // detect errors in kernel execution
    // copy memory back to device (blocks until kernel done)
    checkCudaErrors(cudaMemcpy2D(m_bfr_next.data(), sizeof(State) * m_width, /* pitch==width on host */
                                 d_bfr_next, d_next_pitch,
                                 sizeof(State) * m_width, m_height,
                                 cudaMemcpyDeviceToHost));
    // new gen stored in bfr_next
}
void Life::updateCudaManaged()
{
}
void Life::updateCudaPinned()
{
}
