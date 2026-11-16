# STACK DECISION & ARCHITECTURAL SPECIFICATION
## High Performance Computing (24ISE503) — Activity 01
### Crash-Simulation Workload Decomposition Core

---

## 1. Architectural Stack Decision

| Component | Selected Stack | Rejected Alternatives | Technical Justification |
|---|---|---|---|
| **Core Physics & Time-stepping Engine** | **C++17 + OpenMP (Persistent Parallel Region)** | Per-step Python kernel invocation; Python `multiprocessing` | Eliminates Python runtime and IPC barriers entirely. Single persistent `#pragma omp parallel` loop executes all timesteps with microsecond barrier latency. |
| **Toolchain & Compiler** | **GCC 13.3.0 via WSL2 Ubuntu 24.04 LTS** | Windows TDM-GCC 10.3; Numba on Python 3.14 | TDM-GCC 10.3 was missing `libgomp.spec` (no OpenMP runtime). Numba has no Python 3.14 wheels. GCC 13.3.0 in WSL2 provides native OpenMP 4.5, POSIX thread affinity, and `-O3 -march=native` vectorization. |
| **Orchestration & Pre/Post-Processing** | **CPython 3.14.7 + NumPy 2.5.5** | Pure CLI C++ only | Keeps high-level parametric car mesh generation and Matplotlib visual rendering in Python, interfacing via binary mesh input (`.bin`) and diagnostic time-series output (`.csv`). |
| **Inter-Process / Inter-Language Protocol** | **Fixed-size Binary File Layout (64-byte Header + Contiguous float64/int32 arrays)** | JSON / Protobuf / Python Pickling | Zero parsing overhead; $O(1)$ streaming via `std::ifstream::read` into pre-allocated memory buffers. |

---

## 2. Pipeline Stages (S1–S5 Decomposition)

Every timestep $\Delta t = 5.0\,\mu\text{s}$ executes strictly within one parallel region across $P$ threads:

```
[Start Timestep]
       |
       v
+-----------------------------------------------------------------------------------+
| S1: Spring / Element Force Evaluation (Parallel over M/P springs)                 |
|     - Vectorized Hooke + Euler compressive buckling check (BUCKLE_RATIO = 0.50)   |
|     - Thread-local force scatter-add into thread_forces[tid][i]                   |
|     - Internal elastic strain energy accumulator (E_strain)                       |
+-----------------------------------------------------------------------------------+
       |
    [#pragma omp barrier]
       |
       v
+-----------------------------------------------------------------------------------+
| S2: Force Assembly (Parallel over N/P nodes)                                      |
|     - Sliced reduction: thread_forces[0..P-1][i] summed into force_global[i]      |
|     - 100% contention-free, cache-aligned, zero atomic operations                 |
+-----------------------------------------------------------------------------------+
       |
    [#pragma omp barrier]
       |
       v
+-----------------------------------------------------------------------------------+
| S3: Contact Mechanics (Rigid-Wall Penalty + Spatial-Hash Self-Contact)            |
|     - Planar barrier at X = 0 (Normal = +X)                                       |
|     - Penalty normal force: F_wall = K_wall * (-x_i) - c_wall * v_x (for x_i < 0)  |
|     - Reaction force and external wall work accumulation                          |
+-----------------------------------------------------------------------------------+
       |
    [#pragma omp barrier]
       |
       v
+-----------------------------------------------------------------------------------+
| S4: Numerical Integration (Velocity-Störmer-Verlet, Parallel over N/P nodes)      |
|     - a_i = F_i / m_i - (c_damp / m_i) * v_i                                      |
|     - v_i(t+dt) = v_i(t) + a_i * dt                                               |
|     - x_i(t+dt) = x_i(t) + v_i(t+dt) * dt                                         |
|     - Kinetic energy and viscous dissipation accumulation                         |
+-----------------------------------------------------------------------------------+
       |
    [#pragma omp barrier]
       |
       v
+-----------------------------------------------------------------------------------+
| S5: CFL Reduction & Step Diagnostics (Master Thread / Periodic Reduction)         |
|     - Local critical timestep evaluation: dt_crit = min(L_ij / c_sound)           |
|     - Periodic energy balance check: Drift = |E_tot - E_0| / E_0                  |
+-----------------------------------------------------------------------------------+
       |
    [#pragma omp barrier]
       |
       v
[Next Timestep t + dt]
```

---

## 3. Energy Balance Formulation

The system accounts for conservative and non-conservative energy channels:

$$E_{\text{total}}(t) = E_k(t) + E_s(t) + E_{\text{damp}}(t) + E_{\text{plastic}}(t) - W_{\text{ext}}(t)$$

1. **Kinetic Energy ($E_k$):** $\sum_{i=1}^N \frac{1}{2} m_i \|\mathbf{v}_i\|^2$
2. **Elastic Strain Energy ($E_s$):** $\sum_{m=1}^M \frac{1}{2} k_m (d_m - L_{0, m})^2 \quad (\text{for unbuckled springs } d_m \ge \eta L_{0, m})$
3. **Viscous Dissipation ($E_{\text{damp}}$):** $\sum_{n=1}^{\text{step}} \sum_{i=1}^N c_{\text{damp}} \|\mathbf{v}_i^{(n)}\|^2 \Delta t$
4. **Plastic Buckling Dissipation ($E_{\text{plastic}}$):** Energy dissipated when structural members exceed critical compressive load and buckle plastically.
5. **External Work ($W_{\text{ext}}$):** Work done by the rigid obstacle reaction wall $\int \mathbf{F}_{\text{wall}} \cdot d\mathbf{x}$.

**Numerical Energy Drift Definition:**
$$\text{Drift}(t) = \frac{|E_{\text{total}}(t) - E_{\text{initial}}|}{E_{\text{initial}}}$$

---

## 4. Hardware Affinity & Karp-Flatt Analysis for Heterogeneous Cores

### 4.1 CPU Topology (Intel Core i5-1334U)
- **2 Performance Cores (P-cores):** Cores 0 & 1, 2 threads/core (Hyper-Threading) = Threads 0–3. Fast turbo (4.6 GHz), 2MB L2/core.
- **8 Efficient Cores (E-cores):** Cores 2–9, 1 thread/core = Threads 4–11. Energy-efficient (1.3–3.4 GHz), shared 4MB L2 cluster.
- **Total:** 10 Physical Cores, 12 Logical Execution Threads.

### 4.2 Karp-Flatt Metric Formulation
The Karp-Flatt metric $e$ isolates the parallel efficiency degradation:

$$e = \frac{\frac{1}{S(P)} - \frac{1}{P}}{1 - \frac{1}{P}}$$

- On a **homogeneous** cluster, $e$ is the empirical serial execution fraction $(1 - P)$.
- On an **asymmetric hybrid** processor (like this 2P+8E chip), $e$ increases sharply when $P > 4$ because OpenMP barriers force fast P-core threads to spin-wait for slower E-core threads to finish their spring chunks.

