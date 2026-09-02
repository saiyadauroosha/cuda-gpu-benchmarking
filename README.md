# CUDA GPU Benchmarking

## Project Overview

This repository is a learning-focused CUDA GPU benchmarking project for studying GPU programming and CPU-vs-GPU performance.

The current milestone implements **element-wise vector addition** on both the CPU and an NVIDIA GPU using CUDA.

The project focuses on understanding:

- CUDA kernels and GPU parallelism
- Threads, blocks, and grids
- Host and device memory
- CPU-to-GPU and GPU-to-CPU data transfers
- CPU and GPU timing methods
- Kernel-only performance versus end-to-end GPU performance
- The effect of memory-transfer overhead on GPU acceleration

The goal is not simply to produce faster code, but to understand why CPU and GPU performance differ and to build benchmarks that can be explained clearly in technical discussions.

---

## Current Workload: Vector Addition

Given two input vectors `A` and `B`, the program computes:

```text
C[i] = A[i] + B[i]
```

For the current benchmark:

```text
A = [1, 1, 1, ...]
B = [2, 2, 2, ...]
C = [3, 3, 3, ...]
```

The same workload is executed using both a CPU implementation and a CUDA implementation.

---

## CPU Implementation

The CPU implementation processes the vector sequentially using a C++ loop:

```cpp
for (int i = 0; i < a.size(); i++) {
    c[i] = a[i] + b[i];
}
```

The output vector is allocated before the benchmark begins so repeated dynamic resizing does not affect the measured computation time.

---

## CUDA Implementation

The CUDA version assigns one vector element to each GPU thread.

The kernel is:

```cpp
__global__ void vectorAddGPU(
    const int* a,
    const int* b,
    int* c,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        c[i] = a[i] + b[i];
    }
}
```

Each thread calculates its global index using:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

The thread then processes the corresponding vector element:

```text
Thread 0 -> C[0]
Thread 1 -> C[1]
Thread 2 -> C[2]
...
```

The bounds check:

```cpp
if (i < N)
```

prevents extra launched threads from accessing memory beyond the end of the vector.

---

## CUDA Launch Configuration

The current implementation uses:

```text
256 threads per block
```

The number of blocks is calculated using:

```cpp
int blocksPerGrid =
    (N + threadsPerBlock - 1) / threadsPerBlock;
```

This ceiling-division formula ensures that enough threads are launched even when the vector size is not exactly divisible by 256.

The kernel is launched using:

```cpp
vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(
    d_a,
    d_b,
    d_c,
    N
);
```

---

## CUDA Memory Flow

The input vectors are initially stored in CPU memory.

GPU memory is allocated using `cudaMalloc()`:

```text
CPU memory                    GPU memory

A --------------------------> d_a
B --------------------------> d_b
                               |
                               | CUDA kernel
                               v
                              d_c
                               |
C <---------------------------+
```

The complete flow is:

```text
CPU input vectors
      |
      | cudaMemcpyHostToDevice
      v
GPU input memory
      |
      | CUDA kernel
      v
GPU output memory
      |
      | cudaMemcpyDeviceToHost
      v
CPU output vector
```

GPU memory is released using `cudaFree()` after the benchmark finishes.

---

## Project Structure

```text
cuda-gpu-benchmarking/
|
├── src/
│   ├── cpu/
│   │   └── vector_add.cpp
│   │
│   └── cuda/
│       └── vector_add.cu
│
├── include/
├── benchmarks/
├── scripts/
├── results/
│   └── vector_add_results.txt
│
├── docs/
├── CMakeLists.txt
└── README.md
```

Current implementation:

- `src/cpu/vector_add.cpp` contains the initial CPU vector-addition implementation.
- `src/cuda/vector_add.cu` contains the current CPU-vs-CUDA benchmark.
- `results/vector_add_results.txt` stores the measured benchmark results.

Other directories are reserved for later milestones.

---

# Build Requirements

## Development Environment

The current benchmark was developed and measured using:

- GPU: NVIDIA GeForce MX150
- GPU memory: 2 GB
- NVIDIA Driver: 581.83
- CUDA Toolkit: 12.6
- nvcc: 12.6.85
- g++: 11.4.0
- Operating environment: Ubuntu 22.04 through WSL

`nvidia-smi` reports CUDA 13.0 compatibility from the installed NVIDIA driver, while the CUDA Toolkit actually used to compile this project is CUDA 12.6.

---

## Build

From the project root:

```bash
nvcc -O2 src/cuda/vector_add.cu -o vector_add
```

`-O2` enables compiler optimization for the benchmark build.

---

## Run

The program accepts the vector size as a command-line argument.

Example:

```bash
./vector_add 1000000
```

This runs the benchmark with:

```text
N = 1,000,000 elements
```

If no vector size is provided, the default is:

```text
1,000,000 elements
```

---

# Benchmark Methodology

The benchmark is designed to distinguish between **computation performance** and the **total cost of using the GPU**.

## Workload Sizes

The current benchmark evaluates:

```text
100,000 elements
1,000,000 elements
5,000,000 elements
10,000,000 elements
```

The same input values are used for the CPU and GPU implementations.

---

## Number of Repetitions

Each CPU and GPU measurement is repeated:

```text
10 times
```

The reported value is the arithmetic mean of those 10 measured runs.

Using repeated measurements reduces the influence of noise from individual executions.

---

## CPU Timing

CPU execution time is measured using:

```cpp
std::chrono::steady_clock
```

The timer surrounds only the CPU vector-addition computation.

It does not include input-vector initialization.

Therefore:

```text
CPU time
=
CPU vector-addition computation
```

---

## GPU Warm-Up

Before collecting measured GPU timings, one CUDA kernel is executed as a warm-up:

```cpp
vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(
    d_a,
    d_b,
    d_c,
    N
);

cudaDeviceSynchronize();
```

The warm-up run is not included in the reported benchmark.

This reduces the effect of first-use CUDA initialization overhead on the measured kernel execution.

---

## GPU Kernel Timing

CUDA events are used to measure kernel execution:

```cpp
cudaEventRecord(start);

vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(
    d_a,
    d_b,
    d_c,
    N
);

cudaEventRecord(stop);
cudaEventSynchronize(stop);
```

Therefore:

```text
GPU kernel time
=
CUDA kernel execution only
```

Host-device memory transfers are not included in this measurement.

---

## GPU End-to-End Timing

The end-to-end GPU benchmark includes:

```text
CPU -> GPU copy of A
+
CPU -> GPU copy of B
+
CUDA kernel execution
+
GPU -> CPU copy of C
```

Therefore:

```text
GPU end-to-end time
=
Host-to-device transfers
+
Kernel execution
+
Device-to-host transfer
```

This measurement provides a more realistic view of the total cost of using the GPU when the input data begins in CPU memory and the result is required back on the CPU.

GPU memory allocation with `cudaMalloc()` is performed outside the repeated timed region.

---

## Correctness Check

The CPU and GPU implementations both produce the expected value:

```text
3
```

for the current inputs.

The benchmark currently prints representative CPU/GPU results, including the first and last GPU elements, as a basic correctness check.

A future milestone will add full element-by-element CPU/GPU result validation.

---

# Benchmark Results

Each value below is the average of 10 repetitions.

| Vector Size | Average CPU Time | Average GPU Kernel Time | Average GPU End-to-End Time |
|---:|---:|---:|---:|
| 100,000 | 0.126322 ms | 0.123437 ms | 1.35809 ms |
| 1,000,000 | 1.92214 ms | 0.565603 ms | 7.32592 ms |
| 5,000,000 | 7.53032 ms | 1.92325 ms | 31.3628 ms |
| 10,000,000 | 12.3349 ms | 3.55347 ms | 44.3943 ms |

Raw measurements are also stored in:

```text
results/vector_add_results.txt
```

---

# Results Analysis

The benchmark shows an important distinction between **GPU computation speed** and **overall GPU execution cost**.

For example, at:

```text
N = 1,000,000
```

the measured averages were:

```text
CPU computation:       1.92214 ms
GPU kernel:            0.565603 ms
GPU end-to-end:        7.32592 ms
```

The CUDA kernel performs the vector addition faster than the CPU implementation.

However, the complete GPU operation is slower because the data must first be transferred from CPU memory to GPU memory and the resulting vector must then be transferred back.

The same pattern is visible at larger vector sizes.

For 10,000,000 elements:

```text
CPU computation:       12.3349 ms
GPU kernel:             3.55347 ms
GPU end-to-end:        44.3943 ms
```

The GPU kernel is substantially faster than the CPU computation, but memory-transfer overhead dominates the overall GPU execution time.

---

## Key Observation

A faster CUDA kernel does not automatically mean a faster application.

Vector addition performs very little computation for each element:

```cpp
c[i] = a[i] + b[i];
```

while requiring three large memory movements:

```text
A -> GPU
B -> GPU
C -> CPU
```

As a result, vector addition has relatively low computational intensity, and CPU-GPU transfer overhead can outweigh the benefit of GPU parallelism.

This benchmark therefore demonstrates why GPU performance must be evaluated using both:

```text
kernel-only performance
```

and:

```text
end-to-end application performance
```

---

# Current Learning Outcomes

This milestone provided hands-on experience with:

- CUDA host and device concepts
- `__global__` kernels
- GPU thread indexing
- `threadIdx`
- `blockIdx`
- `blockDim`
- CUDA grids and blocks
- Kernel launch configuration
- Bounds checking in CUDA kernels
- `cudaMalloc`
- `cudaMemcpy`
- `cudaFree`
- Host-to-device transfers
- Device-to-host transfers
- CUDA warm-up runs
- `cudaDeviceSynchronize`
- CUDA event timing
- `std::chrono`
- Repeated benchmark measurements
- Averaging benchmark results
- Compiler optimization with `-O2`
- Kernel-only versus end-to-end timing
- CPU-GPU memory-transfer overhead
- Basic CPU/GPU performance analysis

---


# Matrix Multiplication Benchmark

## Overview

The second workload compares CPU and CUDA implementations of square matrix multiplication.

For two `N x N` matrices `A` and `B`, the result matrix `C` is computed using:

```text
C[row][col] =
sum of A[row][k] * B[k][col]
for k = 0 ... N-1
```

Unlike vector addition, matrix multiplication performs substantially more computation for each output element. This makes it a useful workload for studying when GPU parallelism can outweigh CPU-GPU memory-transfer overhead.

---

## CPU Implementation

The CPU implementation uses three nested loops:

```cpp
for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {

        int sum = 0;

        for (int k = 0; k < N; k++) {
            sum += A[row * N + k]
                 * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}
```

The matrices are stored as one-dimensional vectors, so a matrix element at `(row, col)` is accessed using:

```cpp
row * N + col
```

---

## Naive CUDA Implementation

The current CUDA implementation assigns one output matrix element to each GPU thread.

Each thread determines its row and column using:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

The thread then computes one output element:

```cpp
C[row * N + col]
```

by multiplying one row of `A` with one column of `B`.

The current implementation uses:

```text
16 x 16 threads per block
```

or 256 threads per block.

This implementation is considered the **naive CUDA baseline** because each thread reads matrix values directly from global GPU memory.

---

## Matrix Multiplication Benchmark Methodology

The benchmark evaluates three matrix sizes:

```text
128 x 128
256 x 256
512 x 512
```

Each CPU and GPU measurement is repeated 10 times, and the arithmetic mean is reported.

The CPU measurement includes only CPU matrix-multiplication computation.

The GPU benchmark reports two measurements:

### GPU Kernel Time

Measures only execution of the CUDA matrix-multiplication kernel using CUDA events.

### GPU End-to-End Time

Includes:

```text
CPU -> GPU copy of A
+
CPU -> GPU copy of B
+
CUDA kernel execution
+
GPU -> CPU copy of C
```

GPU memory allocation is performed outside the repeated timed section.

A CUDA warm-up kernel is executed before measured GPU runs to reduce first-launch initialization overhead.

The benchmark is compiled using:

```bash
nvcc -O2 src/cuda/matrix_mul.cu -o matrix_mul
```

The matrix size can be supplied from the command line:

```bash
./matrix_mul 256
```

---

## Matrix Multiplication Results

Each value below is the average of 10 repetitions.

| Matrix Size | Average CPU Time | Average GPU Kernel Time | Average GPU End-to-End Time |
|---:|---:|---:|---:|
| 128 x 128 | 4.20419 ms | 0.286003 ms | 1.22663 ms |
| 256 x 256 | 40.4832 ms | 0.619165 ms | 1.45423 ms |
| 512 x 512 | 714.317 ms | 5.15861 ms | 7.69924 ms |

Raw measurements are stored in:

```text
results/matrix_mul_results.txt
```

---

## Matrix Multiplication Observations

The matrix-multiplication benchmark behaves very differently from vector addition.

With vector addition, the CUDA kernel was faster than the CPU computation, but CPU-GPU memory-transfer overhead caused the end-to-end GPU execution to be slower.

For matrix multiplication, the GPU remains substantially faster even after including memory transfers.

For example, at `512 x 512`:

```text
CPU:                 714.317 ms
GPU kernel:            5.15861 ms
GPU end-to-end:        7.69924 ms
```

Matrix multiplication performs much more computation for each output element than vector addition. This gives the GPU enough parallel work to offset the cost of moving the matrices between CPU and GPU memory.

This demonstrates an important GPU-performance principle:

> GPU acceleration is particularly effective when a workload provides enough parallel computation relative to its data-transfer cost.

The current CUDA kernel is still a baseline implementation. It repeatedly reads matrix elements from global GPU memory. A future milestone will use shared-memory tiling to reduce redundant global-memory accesses.

---

# Future Work

Planned next steps include:

- Implement tiled matrix multiplication using CUDA shared memory.
- Compare naive CUDA matrix multiplication with the tiled implementation.
- Study the effect of shared-memory reuse on kernel performance.
- Add complete CPU/GPU result validation.
- Add CUDA error checking.
- Automate benchmark execution across workload sizes.
- Export benchmark data to CSV.
- Generate CPU-versus-GPU performance plots.

