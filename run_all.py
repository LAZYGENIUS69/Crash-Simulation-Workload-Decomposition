"""
run_all.py — Master Reproducibility Pipeline
============================================
Course  : High Performance Computing (24ISE503), B.Tech ISE Sem V
Activity: 01 (Crash-Simulation Workload Decomposition)
Machine : Intel 12-core, Windows 11, Python 3.14.7

Single-command pipeline to reproduce:
  1. Mesh generation & unit correctness test (Parallel == Sequential to float eps)
  2. Multi-configuration benchmark execution across core counts & problem sizes
  3. Generation of all publication-ready figures for the academic report

Run:
  python run_all.py
"""

import subprocess
import sys
import time
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.resolve()

def run_step(step_name, command):
    print("\n" + "=" * 70)
    print(f"PIPELINE STEP: {step_name}")
    print(f"Executing: {' '.join(command)}")
    print("=" * 70)
    t0 = time.perf_counter()
    res = subprocess.run(command, cwd=BASE_DIR)
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"[-] Step '{step_name}' failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[+] Step '{step_name}' completed successfully in {elapsed:.2f}s")

def main():
    print("#" * 70)
    print("  CRASH-SIMULATION HPC SUBMISSION REPRODUCIBILITY HARNESS")
    print("#" * 70)

    # 1. Correctness Verification
    run_step("Verification: Floating-Point Correctness Check",
             [sys.executable, "test_correctness.py"])

    # 2. Dependency Graph Generation
    run_step("Graph Analysis: Task Dependency & Synchronization Points",
             [sys.executable, "dep_graph.py"])

    # 3. High-Fidelity Multi-Core Benchmarking
    run_step("Benchmark: Multi-Core Scaling & Speedup (Medium & Large Meshes)",
             [sys.executable, "run_benchmark2.py"])

    # 4. Rendering & Visualization
    run_step("Visualizations: 3D Deformations, Domain Decomposition & Amdahl Curves",
             [sys.executable, "visualize.py"])

    print("\n" + "#" * 70)
    print("  ALL ARTIFACTS, FIGURES, AND DATASETS GENERATED SUCCESSFULLY.")
    print("  Figures: results stored in ./figures and ./results")
    print("#" * 70)

if __name__ == "__main__":
    main()

