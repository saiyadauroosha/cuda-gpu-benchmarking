# Project Overview

This repository is a learning-focused CUDA GPU benchmarking project. The current milestone is only a clean scaffold. Future work will compare CPU and CUDA implementations of computational workloads and document what each component does.

# Goals

Planned work:

- Build simple CPU baseline implementations.
- Add CUDA implementations after the project structure is established.
- Measure runtime consistently across CPU and GPU versions.
- Explain enough design and code detail to support technical interview discussion.

# Planned Architecture

Planned work:

- `src/cpu/` will contain CPU implementations.
- `src/cuda/` will contain CUDA implementations.
- `include/` will contain shared headers.
- `benchmarks/` will contain benchmark entry points.
- `scripts/` will contain helper scripts for running or processing benchmarks.
- `results/` will contain measured benchmark output.
- `docs/` will contain methodology and learning notes.

# Build Requirements

Planned work: exact compiler, CUDA, and build requirements will be documented when source code is added.

The current scaffold includes `CMakeLists.txt` for future build configuration, but it does not define build targets yet.

# Benchmark Methodology

Planned work: benchmarks will document workload size, hardware, driver version, CUDA version, command line, and timing method before reporting results.

# Results

Planned work: no benchmark results have been measured yet.

# Future Work

Planned work:

- Add the first CPU-only workload.
- Add a matching CUDA implementation in a later milestone.
- Add repeatable benchmark commands.
- Record measured results without fabricated performance claims.
