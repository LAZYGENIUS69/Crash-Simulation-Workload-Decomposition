# STALE ARTIFACTS NOTE

The files in this directory (`results/stale/`) are historical artifacts generated during Iterations 1 through 4 prior to the discovery and resolution of two critical numerical issues in Iteration 4b:

1. **Non-Conservative Spring Force Kernel:** Prior to Iteration 4b, the Hooke force formulation divided by distance squared ($\mathbf{F} \propto \frac{\Delta \mathbf{x}}{\|\Delta \mathbf{x}\|^2}$ instead of $\frac{\Delta \mathbf{x}}{\|\Delta \mathbf{x}\|}$), causing an artificial energy drift ($\approx 22.3\%$) in large-displacement regimes.
2. **First-Order Euler-Cromer Integration:** Previous iterations used semi-implicit Euler integration rather than a true second-order symplectic Velocity-Verlet scheme.

### Stale Files:
- `benchmark_results.json`: Iteration 1 Python multiprocessing sweep.
- `benchmark2_results.json`: Iteration 2 Python ablation.
- `cpp_benchmark_results.json`: Iteration 3 preliminary C++ OpenMP sweep.
- `iteration4_experiments.json`: Iteration 4 pre-closure experiments.
- `energy_drift.png`: Pre-closure energy drift plot exhibiting non-conservative loss.
- `probe_*.json`: Initial exploratory Python multiprocessing probes.

### Current Verified Results:
Current, fully validated, and energy-closed benchmarks are located directly in `results/`:
- `results/hw.json`: Complete hardware specification, memory hierarchy, and STREAM triad bandwidth.
- `results/energy_balance_term_by_term.csv`: Verified energy conservation data ($0.010\%$ drift).
- `results/energy_closure.png`: High-resolution term-by-term energy closure plot.
- `results/assembly_sweep_raw.csv`: Raw 5-repeat timing data for Modes 0, 1, and 2 across 10k, 100k, and 500k node meshes.
- `results/assembly_scaling_comparison.png`: Scaling curves across thread counts and precision modes.
