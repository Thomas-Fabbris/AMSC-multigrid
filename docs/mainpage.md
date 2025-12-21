# Algebraic Multigrid handson

This repository contains an implementation of Algebraic
Multigrid (AMG) methods and supporting utilities for solving 2D Poisson-like problems using Eigen.

## Key Components

- `AMG` (in `src/AMG.hpp`, `src/AMG.cpp`): builds an AMG hierarchy, computes
  strong connections, constructs interpolation/restriction operators, and
  provides V-cycle solve routines.
- Discretization helpers (in `include/DiscretizeData.hpp`): utilities to build
  finite-difference grids and assemble variable-coefficient Poisson matrices
  and right-hand sides.
- Jacobi solvers (in `include/jacobi.hpp`): sequential and parallel Jacobi
  relaxation routines used as smoothers within multigrid cycles.
- Example driver (`src/example.cpp`, `src/main.cpp`): demonstrates assembly
  of a test problem and usage of the AMG solver.

## Features

- Simple C/F coarsening heuristic based on a greedy degree selection.
- Classical direct interpolation and Galerkin coarse operators.
- Optional OpenMP acceleration for local assembly and smoothing routines.

## Building and Running

From the project root:

```bash
mkdir -p build
cd build
cmake ..
make
./main
```

The example program assembles a test Poisson problem and runs the AMG
solver; outputs and usage are shown by the example executable.

## For Developers

- Sources are under `src/` and `include/`.
- The Doxygen documentation is generated from comments and markdown files in
  `docs/` and the repository `README.md`.
