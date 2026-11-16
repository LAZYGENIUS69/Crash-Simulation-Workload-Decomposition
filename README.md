# Crash-Simulation Workload Decomposition
### Course: High Performance Computing (24ISE503) — Activity 01

Crash-Simulation Workload Decomposition - HPC Experiential Learning Activity 01 (C++17 + OpenMP explicit crash solver, CUDA bonus).

---

## 1. Project Purpose

This repository models an explicit 3D vehicle crash simulation workload and evaluates parallel workload decomposition across multi-core CPUs. The solver simulates impact kinematics, Hookean spring-mass lattice structural mechanics, elastoplastic buckling yield, rigid penalty barrier impact, and spatial-hash self-contact detection. 

We systematically profile parallel scaling across OpenMP thread configurations, evaluate three distinct force assembly strategies (Private Slabs, Atomic Updates, and Node-Centric CSR Gather), measure empirical memory bandwidth against STREAM benchmarks, and perform closed-loop energy conservation audits.

*Note: Repository history was organized from the project ledger after development.*

---

## 2. Toolchain Versions & Environment

- **OS:** Windows 11 Home (Build 26200) running WSL2 Ubuntu 24.04 LTS (Kernel `5.15.167.4-microsoft-standard-WSL2`)
- **C++ Compiler:** GCC 13.3.0 (`-std=c++17 -O3 -fopenmp -march=native`)
- **Build System:** CMake 3.28.3
- **Python Runtime:** Python 3.12 / 3.14 (NumPy 2.2+, SciPy 1.15+, Pandas 2.2+, Matplotlib 3.10+, Pytest 8.3+)

---

## 3. Hardware Architecture Summary

- **Host Processor:** 13th Gen Intel Core i5-1334U (Raptor Lake, Family 6, Model 186)
- **Cores & Topologies:** 10 Physical Cores (2 Performance-cores with Hyper-Threading + 8 Efficient-cores = 12 Threads)
- **L3 Cache:** **12 MiB shared L3 cache** (confirmed via `lscpu`)
- **System Memory:** 16 GB Dual-Channel DDR
- **Peak Sustained Memory Bandwidth:** **35.32 GB/s** (measured via OpenMP STREAM Triad across 4 threads)
- **Virtualization Note:** WSL2 executes under Microsoft Hyper-V, which dynamically multiplexes 12 virtual CPUs across host hardware threads. Hardware core pinning to physical P- vs E-cores is abstracted by the hypervisor.
- Full hardware metadata is logged in [results/hw.json](results/hw.json).

---

## 4. Build and Run Instructions

### A. Building the C++ Solver via CMake (in WSL2 Ubuntu)
```bash
# Clone the repository
git clone https://github.com/LAZYGENIUS69/Crash-Simulation-Workload-Decomposition.git
cd "Crash-Simulation-Workload-Decomposition"

# Configure and compile release targets
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```
This produces three executables in `bin/`:
1. `bin/crash_solver`: Standard FP64 double-precision solver.
2. `bin/crash_solver_fp32`: FP32 single-precision solver (memory bandwidth sensitivity).
3. `bin/stream_bench`: OpenMP STREAM Triad memory bandwidth benchmark.

### B. Running the Solver
```bash
# Syntax:
# ./bin/crash_solver <mesh.bin> <metrics.csv> <final_pos.bin> [threads] [assembly_mode] [sched_mode]
# assembly_mode: 0=Private Slabs, 1=Atomics, 2=Node Gather CSR
# sched_mode   : 0=Static, 1=Dynamic-16, 2=Dynamic-64, 3=Dynamic-256, 4=Guided

# Example: Run 8 threads using Node Gather CSR on a test mesh
./bin/crash_solver tmp_energy/case_b_mesh.bin results/metrics.csv results/final_pos.bin 8 2 0
```

---

## 5. Verification and Test Suite

Fast unit tests verify physical contact mechanics, numerical equivalence, Python vs. C++ baseline parity, and elastic energy conservation:
```bash
# Run fast test suite (pytest excluding long runs)
pytest -v -m "not slow"

# Run complete verification including 1,500-step crash test
pytest -v
```

> **Validation Note:** Validation between the Python reference solver and the C++ solver demonstrates numerical implementation agreement between the two codebases (within $8.16 \times 10^{-9}$ relative error). Physical correctness is independently established by the closed-form harmonic oscillator penalty wall test (0.155% error) and rigorous closed-loop energy conservation audits (residual $\le 0.010\%$).

---

## 6. Key Experimental Results

| Metric | Measured Finding | Evidence File |
| :--- | :--- | :--- |
| **Elastic Energy Conservation** | Drift $\le \mathbf{0.00092\%}$ over 1,000 steps ($E_0 = 32,844.94\text{ J}$) | [results/energy_closure.png](results/energy_closure.png) |
| **Full Crash Energy Closure** | Drift $\le \mathbf{0.010\%}$ across wall impact, damping, & buckling | [results/energy_balance_term_by_term.csv](results/energy_balance_term_by_term.csv) |
| **Assembly Trade-off (100k Nodes)** | Mode 2 (Node Gather CSR) achieves **$2.17\times$ speedup**; Mode 0 private slabs degrades due to L3 cache spill ($> 12\text{ MiB}$) | [results/assembly_scaling_comparison.png](results/assembly_scaling_comparison.png) |
| **Assembly Trade-off (500k Nodes)** | Mode 2 Gather scales to **$2.11\times$ (FP64)** and **$2.37\times$ (FP32)**; Mode 1 Atomics stalls due to bus contention | [results/assembly_sweep_raw.csv](results/assembly_sweep_raw.csv) |
| **Peak Memory Bandwidth** | STREAM Triad saturates at **35.32 GB/s** at $\ge 4$ threads | [results/hw.json](results/hw.json) |
| **Arithmetic Intensity** | Estimated at $\approx 0.38\text{ FLOP/byte}$, consistent with memory-bound behavior | [docs/LEDGER.md](docs/LEDGER.md) |
| **Analytic Contact Validation** | Barrier penetration matches closed-form harmonic oscillator ($0.155\%$ error) | [tests/test_analytic_wall.py](tests/test_analytic_wall.py) |
| **Python vs C++ Baseline Parity** | C++ output matches Python reference solver within $8.16 \times 10^{-9}$ relative; compiled C++ executes in $2.03\text{ ms}$ vs. $31.8\text{ ms}$ in Python ($15.7\times$ speedup) | [tests/test_validate_cpp.py](tests/test_validate_cpp.py) |

---

## 7. Project Documentation

- [docs/LEDGER.md](docs/LEDGER.md): Chronological engineering ledger, test logs, and iteration checkpoints.
- [docs/STACK_DECISION.md](docs/STACK_DECISION.md): Architectural decision record and compiler toolchain justification.
- [docs/VIVA_SHEET.md](docs/VIVA_SHEET.md): HPC viva defence questions, concepts, and derivations.
- [REFERENCES.md](REFERENCES.md): Verified academic citations, DOI index, and asset licenses.
- [ISSUES.md](ISSUES.md): Active issue tracker and technical backlog.
- [results/stale/STALE_NOTE.md](results/stale/STALE_NOTE.md): Documentation of pre-fix historical artifacts.

---

## 8. License

This project is licensed under the [MIT License](LICENSE).

