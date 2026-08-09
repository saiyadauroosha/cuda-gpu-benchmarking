# Repository Guidelines

## Project Purpose

Portfolio and learning repository for CUDA, C++, Git, GitHub, and Codex. Prioritize understanding, interview readiness, and small steps over speed.

## Project Structure & Module Organization

No tracked source tree exists yet. Planned layout:

- `src/`: CUDA/C++ benchmark implementations and shared helpers.
- `include/`: public headers used across benchmarks.
- `tests/`: unit, integration, and smoke tests.
- `scripts/`: setup, profiling, plotting, and result-processing utilities.
- `results/`: generated outputs. Commit only curated examples.
- `docs/`: methodology, hardware notes, and interpretation guidance.

When adding major files or components, explain why they exist and how they fit. Use names like `src/memory_bandwidth.cu`.

## Learning & Agent Collaboration Rules

- Explain the plan before implementing any large feature.
- Prefer small incremental changes that can be committed separately.
- Do not generate an entire finished project at once.
- Teach the purpose of each major file and component.
- Explain CUDA or C++ concepts before using them.
- Keep implementations beginner-readable before optimizing them.
- Keep important logic visible; avoid unnecessary abstractions.
- When changing code, state what changed and why.
- Do not fabricate benchmark numbers or performance claims.
- Prioritize code the owner can explain in an interview over clever or overly compact solutions.

## Build, Test, and Development Commands

No build system exists yet. Document canonical commands in `README.md` when added.

Useful checks:

- `nvcc --version`: verify the CUDA compiler version.
- `nvidia-smi`: inspect GPU model, driver, memory, and running processes.
- `git status --short`: check local changes before committing.

Prefer reproducible commands such as `cmake --build build`, `ctest --test-dir build`, or `pytest`.

## Coding Style & Naming Conventions

Use clear C++/CUDA style with 4-space indentation. Use `.cu` for CUDA, `.cuh` or `.hpp` for headers, and `snake_case.py` for Python.

Use descriptive benchmark and kernel names, such as `run_memory_bandwidth_benchmark` and `copy_kernel`. Centralize constants only when it improves clarity.

## Testing Guidelines

Add tests with the first implementation. Use `tests/` for automated checks and include one CUDA smoke test.

Name tests after behavior, for example `test_memory_bandwidth_smoke.py`. Report benchmark results only when measured, with hardware, driver, CUDA version, command line, and settings.

## Commit & Pull Request Guidelines

No commit history exists yet. Use short, imperative subjects, for example `Add memory bandwidth benchmark`.

Pull requests should include a summary, commands run, hardware used, and relevant outputs or plots. Do not commit secrets, private paths, or large raw benchmark dumps.
