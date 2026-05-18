#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(result));
    exit(1);
  }
  return result;
}

void transposeCPU(const float *in, float *out, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      out[c * rows + r] = in[r * cols + c];
    }
  }
}

void fillInput(float *data, int count) {
  for (int i = 0; i < count; i++) data[i] = (float)(i % 1000) * 0.25f;
}

void postprocess(const char *label, const float *ref, const float *res, int count, float ms, int reps) {
  for (int i = 0; i < count; i++) {
    if (res[i] != ref[i]) {
      printf("  %-30s  FAILED at [%d]: ref=%.1f got=%.1f\n", label, i, ref[i], res[i]);
      return;
    }
  }
  double gb = 2.0 * count * sizeof(float) * reps * 1e-9;
  double secs = ms * 1e-3;
  printf("  %-30s  %8.2f GB/s  %8.4f ms/rep\n", label, gb / secs, ms / reps);
}

__global__ void transposeNaive(float *out, const float *in, int rows, int cols) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    int yy = y + j;
    if (x < cols && yy < rows) out[x * rows + yy] = in[yy * cols + x];
  }
}

__global__ void transposeNoBankConflicts(float *out, const float *in, int rows, int cols) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    int yy = y + j;
    if (x < cols && yy < rows) tile[threadIdx.y + j][threadIdx.x] = in[yy * cols + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    int yy = y + j;
    if (x < rows && yy < cols) out[yy * rows + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}

__global__ void transposeBlockSweep(float *out, const float *in, int rows, int cols) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < cols && y < rows) out[x * rows + y] = in[y * cols + x];
}

float runKernelNaive(const float *h_in, const float *h_ref, float *h_out, int rows, int cols, int reps) {
  int count = rows * cols;
  size_t bytes = (size_t)count * sizeof(float);
  float *d_in, *d_out;
  checkCuda(cudaMalloc(&d_in, bytes));
  checkCuda(cudaMalloc(&d_out, bytes));
  checkCuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE_DIM, BLOCK_ROWS);
  dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);

  cudaEvent_t ev_s, ev_e;
  checkCuda(cudaEventCreate(&ev_s));
  checkCuda(cudaEventCreate(&ev_e));
  float ms = 0.0f;

  transposeNaive<<<grid, block>>>(d_out, d_in, rows, cols);
  checkCuda(cudaEventRecord(ev_s));
  for (int i = 0; i < reps; i++) transposeNaive<<<grid, block>>>(d_out, d_in, rows, cols);
  checkCuda(cudaEventRecord(ev_e));
  checkCuda(cudaEventSynchronize(ev_e));
  checkCuda(cudaEventElapsedTime(&ms, ev_s, ev_e));
  checkCuda(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  postprocess("transposeNaive", h_ref, h_out, count, ms, reps);

  checkCuda(cudaEventDestroy(ev_s));
  checkCuda(cudaEventDestroy(ev_e));
  checkCuda(cudaFree(d_in));
  checkCuda(cudaFree(d_out));
  return ms / reps;
}

float runKernelOptimized(const float *h_in, const float *h_ref, float *h_out, int rows, int cols, int reps) {
  int count = rows * cols;
  size_t bytes = (size_t)count * sizeof(float);
  float *d_in, *d_out;
  checkCuda(cudaMalloc(&d_in, bytes));
  checkCuda(cudaMalloc(&d_out, bytes));
  checkCuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE_DIM, BLOCK_ROWS);
  dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);

  cudaEvent_t ev_s, ev_e;
  checkCuda(cudaEventCreate(&ev_s));
  checkCuda(cudaEventCreate(&ev_e));
  float ms = 0.0f;

  transposeNoBankConflicts<<<grid, block>>>(d_out, d_in, rows, cols);
  checkCuda(cudaEventRecord(ev_s));
  for (int i = 0; i < reps; i++) transposeNoBankConflicts<<<grid, block>>>(d_out, d_in, rows, cols);
  checkCuda(cudaEventRecord(ev_e));
  checkCuda(cudaEventSynchronize(ev_e));
  checkCuda(cudaEventElapsedTime(&ms, ev_s, ev_e));
  checkCuda(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  postprocess("transposeNoBankConflicts", h_ref, h_out, count, ms, reps);

  checkCuda(cudaEventDestroy(ev_s));
  checkCuda(cudaEventDestroy(ev_e));
  checkCuda(cudaFree(d_in));
  checkCuda(cudaFree(d_out));
  return ms / reps;
}

float runKernelUnified(const float *h_in, const float *h_ref, float *h_out, int rows, int cols, int reps) {
  int count = rows * cols;
  size_t bytes = (size_t)count * sizeof(float);
  float *u_in, *u_out;
  checkCuda(cudaMallocManaged(&u_in, bytes));
  checkCuda(cudaMallocManaged(&u_out, bytes));
  for (int i = 0; i < count; i++) u_in[i] = h_in[i];

  int devId;
  checkCuda(cudaGetDevice(&devId));
  cudaMemLocation devLoc{cudaMemLocationTypeDevice, devId};
  cudaMemLocation hostLoc{cudaMemLocationTypeHost, 0};
  checkCuda(cudaMemPrefetchAsync(u_in, bytes, devLoc, 0, 0));
  checkCuda(cudaMemPrefetchAsync(u_out, bytes, devLoc, 0, 0));
  checkCuda(cudaDeviceSynchronize());

  dim3 block(TILE_DIM, BLOCK_ROWS);
  dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);

  cudaEvent_t ev_s, ev_e;
  checkCuda(cudaEventCreate(&ev_s));
  checkCuda(cudaEventCreate(&ev_e));
  float ms = 0.0f;

  transposeNoBankConflicts<<<grid, block>>>(u_out, u_in, rows, cols);
  checkCuda(cudaEventRecord(ev_s));
  for (int i = 0; i < reps; i++) transposeNoBankConflicts<<<grid, block>>>(u_out, u_in, rows, cols);
  checkCuda(cudaEventRecord(ev_e));
  checkCuda(cudaEventSynchronize(ev_e));
  checkCuda(cudaEventElapsedTime(&ms, ev_s, ev_e));

  checkCuda(cudaMemPrefetchAsync(u_out, bytes, hostLoc, 0, 0));
  checkCuda(cudaDeviceSynchronize());
  for (int i = 0; i < count; i++) h_out[i] = u_out[i];
  postprocess("transposeUnified", h_ref, h_out, count, ms, reps);

  checkCuda(cudaEventDestroy(ev_s));
  checkCuda(cudaEventDestroy(ev_e));
  checkCuda(cudaFree(u_in));
  checkCuda(cudaFree(u_out));
  return ms / reps;
}

void benchmarkBlockSizes(const float *h_in, int rows, int cols, int reps) {
  int count = rows * cols;
  size_t bytes = (size_t)count * sizeof(float);
  float *d_in, *d_out;
  checkCuda(cudaMalloc(&d_in, bytes));
  checkCuda(cudaMalloc(&d_out, bytes));
  checkCuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  int blocks[] = {8, 16, 32};
  printf("  %-30s  %10s  %12s\n", "Block size", "Bandwidth", "ms/rep");
  for (int k = 0; k < 3; k++) {
    int bs = blocks[k];
    dim3 block(bs, bs);
    dim3 grid((cols + bs - 1) / bs, (rows + bs - 1) / bs);
    cudaEvent_t ev_s, ev_e;
    checkCuda(cudaEventCreate(&ev_s));
    checkCuda(cudaEventCreate(&ev_e));
    float ms = 0.0f;
    transposeBlockSweep<<<grid, block>>>(d_out, d_in, rows, cols);
    checkCuda(cudaEventRecord(ev_s));
    for (int i = 0; i < reps; i++) transposeBlockSweep<<<grid, block>>>(d_out, d_in, rows, cols);
    checkCuda(cudaEventRecord(ev_e));
    checkCuda(cudaEventSynchronize(ev_e));
    checkCuda(cudaEventElapsedTime(&ms, ev_s, ev_e));
    double gb = 2.0 * count * sizeof(float) * reps * 1e-9;
    double secs = ms * 1e-3;
    printf("  %-30s  %8.2f GB/s  %8.4f ms/rep\n", (bs == 8 ? "8x8" : (bs == 16 ? "16x16" : "32x32")), gb / secs, ms / reps);
    checkCuda(cudaEventDestroy(ev_s));
    checkCuda(cudaEventDestroy(ev_e));
  }

  checkCuda(cudaFree(d_in));
  checkCuda(cudaFree(d_out));
}

void benchmark(int rows, int cols, int reps) {
  printf("\n=== %d x %d matrix ===\n", rows, cols);
  int count = rows * cols;
  size_t bytes = (size_t)count * sizeof(float);

  float *h_in = (float *)malloc(bytes);
  float *h_ref = (float *)malloc(bytes);
  float *h_out = (float *)malloc(bytes);

  fillInput(h_in, count);

  clock_t t0 = clock();
  transposeCPU(h_in, h_ref, rows, cols);
  clock_t t1 = clock();
  printf("  %-30s  %8.2f ms\n", "CPU", 1000.0 * (double)(t1 - t0) / CLOCKS_PER_SEC);

  runKernelNaive(h_in, h_ref, h_out, rows, cols, reps);
  runKernelOptimized(h_in, h_ref, h_out, rows, cols, reps);
  runKernelUnified(h_in, h_ref, h_out, rows, cols, reps);

  printf("\n  Block size sensitivity (%d x %d)\n", rows, cols);
  benchmarkBlockSizes(h_in, rows, cols, reps);

  free(h_in);
  free(h_ref);
  free(h_out);
}

int main(int argc, char **argv) {
  int reps = 100;
  if (argc > 1) reps = atoi(argv[1]);
  if (reps < 1) reps = 1;

  int devId;
  checkCuda(cudaGetDevice(&devId));
  cudaDeviceProp props;
  checkCuda(cudaGetDeviceProperties(&props, devId));
  printf("GPU: %s\n", props.name);
  printf("TILE_DIM=%d  BLOCK_ROWS=%d  REPS=%d\n", TILE_DIM, BLOCK_ROWS, reps);

  benchmark(2048, 1024, reps);
  benchmark(4096, 2048, reps);
  benchmark(8192, 4096, reps);

  return 0;
}
