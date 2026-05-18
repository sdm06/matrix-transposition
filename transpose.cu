#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
      std::ostringstream oss__;                                                  \
      oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "        \
            << cudaGetErrorString(err__);                                        \
      throw std::runtime_error(oss__.str());                                     \
    }                                                                            \
  } while (0)

struct Size2D {
  int rows;
  int cols;
};

struct BenchResult {
  int rows;
  int cols;
  double cpu_ms;
  double naive_total_ms;
  double naive_kernel_ms;
  double tiled_total_ms;
  double tiled_kernel_ms;
  double unified_total_ms;
  double unified_kernel_ms;
  double max_abs_err_naive;
  double max_abs_err_tiled;
  double max_abs_err_um;
};

static constexpr int kDefaultIters = 10;

void prefetchToDevice(void *ptr, size_t bytes, int device) {
#if CUDART_VERSION >= 12000
  cudaMemLocation location{};
  location.type = cudaMemLocationTypeDevice;
  location.id = device;
  CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, location, 0, 0));
#else
  CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, device, 0));
#endif
}

void prefetchToHost(void *ptr, size_t bytes) {
#if CUDART_VERSION >= 12000
  cudaMemLocation location{};
  location.type = cudaMemLocationTypeHost;
  location.id = 0;
  CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, location, 0, 0));
#else
  CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, 0));
#endif
}

void cpuTranspose(const float *in, float *out, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      out[c * rows + r] = in[r * cols + c];
    }
  }
}

__global__ void transposeNaive(const float *in, float *out, int rows, int cols) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < cols && y < rows) {
    out[x * rows + y] = in[y * cols + x];
  }
}

template <int TILE_DIM>
__global__ void transposeTiled(const float *in, float *out, int rows, int cols) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  if (x < cols && y < rows) {
    tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
  }

  __syncthreads();

  int tx = blockIdx.y * TILE_DIM + threadIdx.x;
  int ty = blockIdx.x * TILE_DIM + threadIdx.y;

  if (tx < rows && ty < cols) {
    out[ty * rows + tx] = tile[threadIdx.x][threadIdx.y];
  }
}

double maxAbsError(const std::vector<float> &a, const std::vector<float> &b) {
  double max_err = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    max_err = std::max(max_err, static_cast<double>(std::fabs(a[i] - b[i])));
  }
  return max_err;
}

void fillRandom(std::vector<float> &v) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float &x : v) {
    x = dist(rng);
  }
}

double runCpu(const std::vector<float> &in, std::vector<float> &out, int rows,
              int cols) {
  auto t0 = std::chrono::high_resolution_clock::now();
  cpuTranspose(in.data(), out.data(), rows, cols);
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

template <typename KernelFn>
double runKernelTimed(KernelFn &&kernel, dim3 grid, dim3 block,
                      int warmup_iters, int iters) {
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warmup_iters; ++i) {
    kernel(grid, block);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    kernel(grid, block);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return static_cast<double>(ms) / iters;
}

template <int TILE_DIM>
BenchResult runBenchForSize(const Size2D size, int iters) {
  const int rows = size.rows;
  const int cols = size.cols;
  const size_t n_in = static_cast<size_t>(rows) * cols;
  const size_t n_out = static_cast<size_t>(cols) * rows;
  const size_t bytes = n_in * sizeof(float);

  std::vector<float> h_in(n_in);
  std::vector<float> h_cpu(n_out);
  std::vector<float> h_naive(n_out);
  std::vector<float> h_tiled(n_out);
  std::vector<float> h_um(n_out);
  fillRandom(h_in);

  double cpu_ms = runCpu(h_in, h_cpu, rows, cols);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  dim3 block(TILE_DIM, TILE_DIM);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

  auto naive_kernel_runner = [&](dim3 g, dim3 b) {
    transposeNaive<<<g, b>>>(d_in, d_out, rows, cols);
  };
  auto tiled_kernel_runner = [&](dim3 g, dim3 b) {
    transposeTiled<TILE_DIM><<<g, b>>>(d_in, d_out, rows, cols);
  };

  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  auto t0 = std::chrono::high_resolution_clock::now();
  double naive_kernel_ms =
      runKernelTimed(naive_kernel_runner, grid, block, 2, iters);
  CUDA_CHECK(cudaMemcpy(h_naive.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  auto t1 = std::chrono::high_resolution_clock::now();
  double naive_total_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
  auto t2 = std::chrono::high_resolution_clock::now();
  double tiled_kernel_ms =
      runKernelTimed(tiled_kernel_runner, grid, block, 2, iters);
  CUDA_CHECK(cudaMemcpy(h_tiled.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  auto t3 = std::chrono::high_resolution_clock::now();
  double tiled_total_ms =
      std::chrono::duration<double, std::milli>(t3 - t2).count() / iters;

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  float *um_in = nullptr;
  float *um_out = nullptr;
  CUDA_CHECK(cudaMallocManaged(&um_in, bytes));
  CUDA_CHECK(cudaMallocManaged(&um_out, bytes));
  std::copy(h_in.begin(), h_in.end(), um_in);

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  prefetchToDevice(um_in, bytes, device);
  prefetchToDevice(um_out, bytes, device);
  CUDA_CHECK(cudaDeviceSynchronize());

  auto um_kernel_runner = [&](dim3 g, dim3 b) {
    transposeTiled<TILE_DIM><<<g, b>>>(um_in, um_out, rows, cols);
  };

  auto t4 = std::chrono::high_resolution_clock::now();
  double um_kernel_ms = runKernelTimed(um_kernel_runner, grid, block, 2, iters);
  prefetchToHost(um_out, bytes);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::copy(um_out, um_out + n_out, h_um.begin());
  auto t5 = std::chrono::high_resolution_clock::now();
  double unified_total_ms =
      std::chrono::duration<double, std::milli>(t5 - t4).count() / iters;

  CUDA_CHECK(cudaFree(um_in));
  CUDA_CHECK(cudaFree(um_out));

  BenchResult r{};
  r.rows = rows;
  r.cols = cols;
  r.cpu_ms = cpu_ms;
  r.naive_total_ms = naive_total_ms;
  r.naive_kernel_ms = naive_kernel_ms;
  r.tiled_total_ms = tiled_total_ms;
  r.tiled_kernel_ms = tiled_kernel_ms;
  r.unified_total_ms = unified_total_ms;
  r.unified_kernel_ms = um_kernel_ms;
  r.max_abs_err_naive = maxAbsError(h_cpu, h_naive);
  r.max_abs_err_tiled = maxAbsError(h_cpu, h_tiled);
  r.max_abs_err_um = maxAbsError(h_cpu, h_um);
  return r;
}

template <int TILE_DIM>
double runTiledKernelOnlyForBlockSize(const Size2D size, int iters) {
  const int rows = size.rows;
  const int cols = size.cols;
  const size_t n = static_cast<size_t>(rows) * cols;
  const size_t bytes = n * sizeof(float);

  std::vector<float> h_in(n);
  fillRandom(h_in);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE_DIM, TILE_DIM);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

  auto runner = [&](dim3 g, dim3 b) {
    transposeTiled<TILE_DIM><<<g, b>>>(d_in, d_out, rows, cols);
  };
  double ms = runKernelTimed(runner, grid, block, 2, iters);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return ms;
}

std::string speedup(double baseline, double optimized) {
  std::ostringstream oss;
  if (optimized <= 0.0) {
    oss << "n/a";
  } else {
    oss << std::fixed << std::setprecision(2) << (baseline / optimized) << "x";
  }
  return oss.str();
}

int main(int argc, char **argv) {
  try {
    int iters = kDefaultIters;
    if (argc > 1) {
      iters = std::max(1, std::stoi(argv[1]));
    }

    std::vector<Size2D> sizes{
        {2048, 1024},
        {4096, 2048},
        {8192, 4096},
    };

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Iterations per kernel measurement: " << iters << "\n\n";

    std::vector<BenchResult> results;
    results.reserve(sizes.size());
    for (const auto &s : sizes) {
      results.push_back(runBenchForSize<16>(s, iters));
    }

    std::cout << "## Results (ms)\n";
    std::cout << "| Matrix (rows x cols) | CPU | Naive GPU total | Naive GPU kernel | "
                 "Tiled GPU total | Tiled GPU kernel | Unified total | Unified "
                 "kernel | CPU/Tiled(kernel) |\n";
    std::cout << "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto &r : results) {
      std::cout << "|" << r.rows << "x" << r.cols << "|"
                << std::fixed << std::setprecision(3) << r.cpu_ms << "|"
                << r.naive_total_ms << "|" << r.naive_kernel_ms << "|"
                << r.tiled_total_ms << "|" << r.tiled_kernel_ms << "|"
                << r.unified_total_ms << "|" << r.unified_kernel_ms << "|"
                << speedup(r.cpu_ms, r.tiled_kernel_ms) << "|\n";
    }

    std::cout << "\n## Correctness (max |CPU-GPU|)\n";
    std::cout << "| Matrix | Naive | Tiled | Unified |\n";
    std::cout << "|---|---:|---:|---:|\n";
    for (const auto &r : results) {
      std::cout << "|" << r.rows << "x" << r.cols << "|"
                << std::scientific << std::setprecision(3) << r.max_abs_err_naive
                << "|" << r.max_abs_err_tiled << "|" << r.max_abs_err_um << "|\n";
    }

    Size2D block_test_size{4096, 2048};
    std::cout << "\n## Block size sensitivity (kernel only, "
              << block_test_size.rows << "x" << block_test_size.cols << ")\n";
    std::cout << "| Block size | Tiled kernel ms |\n";
    std::cout << "|---|---:|\n";
    std::cout << "|8x8|" << std::fixed << std::setprecision(3)
              << runTiledKernelOnlyForBlockSize<8>(block_test_size, iters) << "|\n";
    std::cout << "|16x16|" << runTiledKernelOnlyForBlockSize<16>(block_test_size, iters)
              << "|\n";
    std::cout << "|32x32|" << runTiledKernelOnlyForBlockSize<32>(block_test_size, iters)
              << "|\n";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
