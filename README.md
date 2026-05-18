# CUDA Matrix Transposition

This project implements matrix transposition in three ways:

1. **CPU baseline** (`cpuTranspose`)
2. **Naive CUDA kernel** (`transposeNaive`)
3. **Optimized CUDA kernel** with tiled shared memory and unified memory (`transposeTiled`)

It benchmarks all implementations, validates correctness, and measures block-size impact.

## Requirements

- NVIDIA GPU + CUDA Toolkit
- `nvcc` compiler

## Build

```bash
nvcc -O3 -std=c++17 -o transpose transpose.cu
```

## Run

```bash
./transpose
```

Optional: pass iteration count for kernel timing averages.

```bash
./transpose 20
```

## What is measured

For each matrix size (`2048x1024`, `4096x2048`, `8192x4096`):

- CPU transpose time
- Naive GPU transpose:
  - kernel-only time
  - total time (transfer + kernel + copy-back)
- Tiled GPU transpose:
  - kernel-only time
  - total time
- Unified memory transpose:
  - kernel-only time
  - total time (with prefetch)
- Correctness against CPU output (max absolute error)

Block-size sensitivity is also reported for tiled kernel (`8x8`, `16x16`, `32x32`).

## Performance results

Measured on:

- GPU: **NVIDIA GeForce RTX 4060 Ti**
- Command: `./transpose 5`

### Results (ms)

| Matrix (rows x cols) | CPU | Naive GPU total | Naive GPU kernel | Tiled GPU total | Tiled GPU kernel | Unified total | Unified kernel | CPU/Tiled(kernel) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048x1024 | 6.568 | 4.199 | 0.050 | 0.187 | 0.016 | 0.310 | 0.016 | 416.49x |
| 4096x2048 | 73.414 | 0.929 | 0.274 | 0.925 | 0.261 | 1.213 | 0.272 | 281.29x |
| 8192x4096 | 393.709 | 3.576 | 1.034 | 3.591 | 1.067 | 4.911 | 1.051 | 368.99x |

### Correctness (max \|CPU-GPU\|)

| Matrix | Naive | Tiled | Unified |
|---|---:|---:|---:|
| 2048x1024 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 4096x2048 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 8192x4096 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

### Block size sensitivity (kernel only)

| Block size | Tiled kernel ms |
|---|---:|
| 8x8 | 0.260 |
| 16x16 | 0.256 |
| 32x32 | 0.274 |

## Analysis

- **CPU vs GPU:** GPU kernel performance is dramatically faster, and advantage grows with larger matrices due to higher parallel throughput.
- **Naive vs tiled CUDA:** On this GPU, tiled and naive kernel times are close for large matrices, but tiled improves memory behavior and is consistently competitive.
- **Unified memory:** Unified memory kernel time is similar to explicit-copy kernel time; total time is slightly higher because of migration/prefetch overhead.
- **Block size impact:** `16x16` is best in this run, slightly better than `8x8` and `32x32`.

## Interpretation notes

- **CPU vs GPU:** GPU should outperform CPU more as matrix size grows because of higher parallelism.
- **Naive vs tiled CUDA:** tiled shared memory improves memory coalescing and reduces global-memory penalties.
- **Unified memory:** simplifies memory handling; performance depends on page migration behavior, mitigated with prefetch.
- **Block size impact:** `16x16` or `32x32` is commonly best, but optimal choice depends on GPU architecture (occupancy, register/shared-memory pressure).

## Potential further optimizations

- Use asynchronous streams and overlap transfer with compute.
- Tune tile dimensions per architecture.
- Add in-place transpose variants (square matrices) with diagonal-block strategies.
- Use pinned host memory for faster transfers in explicit-copy versions.
