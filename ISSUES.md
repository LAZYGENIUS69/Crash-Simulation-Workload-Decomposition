# Project Issues & Technical Backlog

Repository: `LAZYGENIUS69/Crash-Simulation-Workload-Decomposition`

---

### Issue 1: Stale artifacts management following force kernel repair
- **Status:** Resolved in Iteration 4b / Documented
- **Description:** Previous runs (Iterations 1–4) generated speedup and energy figures using a force kernel with a duplicate distance denominator and semi-implicit Euler integration. All pre-fix outputs have been moved to `results/stale/` with `STALE_NOTE.md` explaining the invalidation. Only post-closure data are presented.

### Issue 2: Empirical achieved memory bandwidth vs. STREAM benchmark
- **Status:** Open (Instrumentation backlog)
- **Description:** Current analysis estimates arithmetic intensity at $\approx 0.38\text{ FLOP/byte}$ and shows scaling behavior consistent with memory-bound limits (STREAM ceiling 35.32 GB/s). To make definitive claims, solver profiling must measure hardware performance counters (bytes moved per second, cache misses) or achieved GB/s per run.

### Issue 3: 12-thread performance collapse and OMP_WAIT_POLICY evaluation
- **Status:** Open (Iteration 5/6 evaluation)
- **Description:** Across all assembly modes, running 12 threads causes significant throughput degradation compared to 8 threads. Investigate whether OpenMP spin-waiting creates artificial CPU contention on hyperthreads by testing `OMP_WAIT_POLICY=passive` and evaluating thread barrier sleep intervals.

### Issue 4: Hyper-V virtualization abstraction of physical P/E-core topology
- **Status:** Documented limitation
- **Description:** Under WSL2, Microsoft Hyper-V presents 12 uniform vCPUs with dynamic thread scheduling across the host's 2P+8E cores. Microbenchmarks show $<10\%$ throughput variation across vCPUs. Testing bare-metal core affinity requires native Windows API execution or remains a documented hypervisor constraint.

### Issue 5: Authentic 3D vehicle-front mesh and crumple-zone partitioning
- **Status:** Scheduled for Iteration 5
- **Description:** Transition from the current synthetic rectangular lattice to an authentic procedural vehicle front structure featuring a bumper beam, crash boxes, longitudinal rails with progressive buckling triggers, subframe, and rigid engine block core.

### Issue 6: Outer triangular skin binding and visual deformation rendering
- **Status:** Scheduled for Iteration 5
- **Description:** Generate a 2D triangular surface mesh mapped to the 3D spring lattice nodes to render a visually recognizable automotive body. Produce crash sequence frames, partition domain coloring, and animation.

### Issue 7: GPU acceleration bonus on Google Colab / Kaggle
- **Status:** Scheduled for Iteration 6
- **Description:** Prototype a CUDA / CuPy parallel spring force evaluation kernel on an NVIDIA Tesla GPU in Google Colab to demonstrate heterogeneous acceleration and compare CPU vs. GPU throughput.

### Issue 8: Report condensation to strictly 4–6 pages
- **Status:** Scheduled for Iteration 7
- **Description:** The preliminary report draft is approximately 10 pages. Condense prose, downscale figures, combine benchmark tables, and ensure all 12 required university sections strictly satisfy the 4–6 page limit.
