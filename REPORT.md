# High Performance Computing (24ISE503) — Activity 01
# Crash-Simulation Workload Decomposition

**Course:** B.Tech Information Science and Engineering, Semester V  
**Module:** High Performance Computing (24ISE503)  
**Activity:** Activity 01 (Automotive Domain)  
**Title:** Crash-Simulation Workload Decomposition  
**Author / Student ID:** 24ISE503 Student Submission  
**Date of Experimentation:** November 16, 2026  
**Submission Deadline:** November 21, 2026  

---

## Abstract

Finite element crashworthiness simulation is one of the most computationally demanding High Performance Computing (HPC) workloads in the automotive sector. Real-world vehicle crash solvers such as LS-DYNA, RADIOSS, and PAM-CRASH rely on explicit time integration, requiring millions of spatial degrees of freedom to be updated over tens of thousands of microsecond-scale timesteps under strict Courant-Friedrichs-Lewy (CFL) stability bounds. This study formulates, implements, benchmarks, and analyzes a parallel explicit dynamic crash simulation workload using shared-memory domain decomposition. We construct a 3D structural lattice model of an automotive vehicle comprising up to 30,000 point-mass nodes and 142,135 structural spring-damper elements with localized front crumple zones ($k = 5 \times 10^5\,\text{N/m}$), rigid safety cabin compartments ($k = 5 \times 10^6\,\text{N/m}$), and non-linear compressive buckling. 

The workload is partitioned across CPU cores using a zero-copy shared-memory architecture (`multiprocessing.shared_memory`), completely avoiding inter-process serialization bottlenecks while mapping to cache-coherent Symmetric Multiprocessing (SMP) supercomputer nodes. We construct a complete task dependency graph delineating concurrent force computations, synchronization barriers, and serial reduction phases. Benchmarking across core counts ($P \in \{1, 2, 4, 6, 8, 12\}$) across two workload scales ($11,520$ nodes / $53,692$ springs and $30,000$ nodes / $142,135$ springs) achieves a peak parallel speedup of $2.52\times$ at 6 workers ($77.0\%$ efficiency at 2 workers). Fitting to Amdahl’s Law reveals an empirical parallel fraction $P_{\text{Amdahl}} \approx 72.5\%$, demonstrating that per-step barrier synchronization and global reduction dominate the serial bottleneck ($1 - P \approx 27.5\%$). Numerical correctness is rigorously proven against sequential execution with a maximum relative coordinate discrepancy bounded at $2.13 \times 10^{-16}$ (machine epsilon). We conclude with industrial scalability recommendations, MPI/CUDA cluster extension architectures, and fault tolerance strategies.

---

## 1. Introduction and Industry Background

### 1.1 The Role of Computational Crashworthiness in Modern Automotive Engineering
In the contemporary automotive industry, physical crash testing of physical prototypes is prohibitively expensive, costing upwards of \$250,000 to \$500,000 per vehicle destruction test, while demanding months of fabrication lead time. Consequently, Original Equipment Manufacturers (OEMs) such as BMW, Mercedes-Benz, General Motors, and Toyota rely heavily on Computer-Aided Engineering (CAE) and High Performance Computing to simulate frontal, offset, side-impact, and rollover collisions. Computational crash simulations evaluate structural integrity, energy dissipation within sacrificial crumple zones, cabin intrusion prevention, and passenger biomechanics against regulatory safety protocols (e.g., Euro NCAP, US NHTSA).

```
+-----------------------------------------------------------------------------------+
|                            Typical OEM Crash Simulation Metrics                   |
+--------------------------+-----------------------+--------------------------------+
| Element Count            | 5,000,000 - 20,000,000 elements                        |
| Simulated Real Time      | 80 - 150 milliseconds (crash duration)                 |
| Numerical Time Step      | 0.5 - 2.0 microseconds (CFL bound)                    |
| Total Solver Timesteps   | 50,000 - 200,000 iterations                           |
| Typical Compute Cluster  | 256 - 2048 CPU cores / Multi-GPU HPC nodes             |
| Turnaround Requirement   | Overnight run (< 8 to 12 hours)                        |
+--------------------------+-----------------------+--------------------------------+
```

### 1.2 Mathematical Foundations: Explicit vs. Implicit Dynamics
Dynamic mechanical equilibrium is governed by the second-order partial differential equation of continuum elastodynamics:

$$\mathbf{M} \ddot{\mathbf{u}}(t) + \mathbf{C} \dot{\mathbf{u}}(t) + \mathbf{K}(\mathbf{u}) \mathbf{u}(t) = \mathbf{F}_{\text{ext}}(t)$$

where $\mathbf{M}$ is the mass matrix, $\mathbf{C}$ is the damping matrix, $\mathbf{K}(\mathbf{u})$ is the non-linear internal stiffness operator, $\mathbf{u}(t)$ is the nodal displacement field, and $\mathbf{F}_{\text{ext}}$ represents external contact forces and boundary conditions.

In quasi-static structural analysis, implicit time integration (e.g., Newmark-$\beta$ or Newton-Raphson schemes) is favored because it is unconditionally stable for large timesteps. However, crashworthiness is characterized by extreme geometric discontinuities, material plasticity, contact impact, and buckling. Inverting large non-linear tangential stiffness matrices ($\mathbf{K}_{\text{tangent}} \Delta \mathbf{u} = \mathbf{R}$) at each timestep requires solving immense linear systems ($O(N^3)$ dense or $O(N^{1.5})$ sparse), creating severe inter-node communication latency in parallel environments.

Commercial crash codes (e.g., Ansys LS-DYNA, Altair Radioss) thus exclusively utilize **explicit central-difference time integration** or **Velocity-Störmer-Verlet algorithms**. Assuming a lumped (diagonal) mass matrix $\mathbf{M}$, the nodal acceleration vector at timestep $n$ is decoupled:

$$\mathbf{a}_i^{(n)} = \mathbf{M}_{ii}^{-1} \left( \mathbf{F}_{\text{ext}, i}^{(n)} - \mathbf{F}_{\text{int}, i}^{(n)} - \mathbf{C}_{ii} \mathbf{v}_i^{(n)} \right)$$

Because $\mathbf{M}$ is diagonal, no global matrix inversion is required. Computation scales strictly as $O(N)$, which is ideal for massive parallelization across CPU cores and GPU streaming multiprocessors.

### 1.3 The Courant-Friedrichs-Lewy (CFL) Condition
The fundamental trade-off of explicit integration is conditional numerical stability. The numerical timestep $\Delta t$ cannot exceed the critical transit time of an acoustic dilational wave across the smallest element in the mesh:

$$\Delta t \le \Delta t_{\text{crit}} = \frac{L_{\min}}{c} = L_{\min} \sqrt{\frac{\rho}{E}}$$

where $L_{\min}$ is the minimum characteristic element length, $c$ is the acoustic speed of sound in the structural material, $E$ is Young's modulus, and $\rho$ is density. For structural automotive steel ($E \approx 210\,\text{GPa}$, $\rho \approx 7850\,\text{kg/m}^3$, $c \approx 5170\,\text{m/s}$) with $5\,\text{mm}$ elements, the critical timestep is:

$$\Delta t_{\text{crit}} \approx \frac{0.005\,\text{m}}{5170\,\text{m/s}} \approx 0.967\,\mu\text{s}$$

A $100\,\text{ms}$ physical crash event requires over $100,000$ sequential timesteps. Because every timestep depends on the state of the preceding timestep, parallelization across time is impossible; **parallelism must be extracted entirely across the spatial domain**.

---

## 2. Problem Statement and Objectives

### 2.1 Problem Statement
While explicit dynamic calculations are embarrassingly parallel within an isolated element evaluation phase, full-vehicle crash simulations present severe HPC scaling bottlenecks:
1. **Pervasive Synchronization:** The CFL condition mandates that all nodal coordinates and boundary conditions be globally synchronized at every microsecond step.
2. **Dynamic Workload Imbalance:** Deformation is highly localized at the structural impact interface. Elements in the crumple zone experience extreme compressive strains, contact searches, and plasticity iterations, while nodes in the rear passenger cabin remain purely elastic, creating core starvation in static decompositions.
3. **Communication-to-Computation Inversion:** As core counts grow, the sub-domain assigned to each core shrinks. When element processing time drops below the inter-core communication and barrier latency, parallel efficiency plummets.

### 2.2 Project Objectives
1. **Procedural Vehicle Discretization:** Formulate an original, geometrically recognizable 3D vehicle model comprising thousands of nodal masses and structural spring elements, segmented into crumple zones and cabin safety cells.
2. **Workload Decomposition & Parallel Implementation:** Design and implement both a sequential baseline and a scalable multi-core parallel prototype in Python using POSIX shared-memory architectures, eliminating runtime data marshalling.
3. **Task Dependency Analysis:** Construct an explicit task dependency graph that formally identifies concurrent tasks, precedence constraints, and critical synchronization barriers.
4. **Empirical Benchmarking & Amdahl Validation:** Conduct rigorous performance profiling across multiple core counts ($P \in \{1, 2, 4, 6, 8, 12\}$) and workload sizes ($11.5\text{k}$ and $30\text{k}$ nodes), quantifying wall-clock runtime, speedup, efficiency, and load imbalance ratios.
5. **Architectural & Industrial Synthesis:** Analyze serial fractions using Amdahl's and Gustafson's scaling laws, verify floating-point correctness, and propose concrete production recommendations for distributed-memory MPI clusters and GPU accelerators.

---

## 3. Methodology and Architecture

### 3.1 3D Structural Vehicle Mesh Model
To eliminate intellectual property constraints while retaining realistic structural mechanics, we developed a procedural 3D automotive lattice. The vehicle envelope features realistic proportional dimensions: Length $L = 4.5\,\text{m}$, Width $W = 1.8\,\text{m}$, and Height $H = 1.5\,\text{m}$, with a sloped front hood profile, flat cabin roof, and tapered trunk.

The structure is divided longitudinally into three distinct physical zones:
1. **Front Crumple Zone ($0 \le X / L \le 0.20$):** Low structural stiffness ($k_{\text{soft}} = 5 \times 10^5\,\text{N/m}$) designed to buckle plastically and absorb kinetic energy.
2. **Passenger Safety Cabin ($0.20 < X / L \le 0.80$):** High structural stiffness ($k_{\text{stiff}} = 5 \times 10^6\,\text{N/m}$) engineered to resist intrusion and preserve survival space.
3. **Rear Vehicle Zone ($0.80 < X / L \le 1.00$):** Moderate stiffness with fixed boundary constraints representing rigid axle and barrier reaction points.

```
+-----------------------------------------------------------------------------------+
|                     Schematic of 3D Vehicle Structural Zoning                     |
+--------------------------+---------------------------------+----------------------+
|   FRONT CRUMPLE ZONE     |      CABIN / SAFETY CELL        |  REAR CRUMPLE / CLAMP|
|       (0% - 20%)         |          (20% - 80%)            |     (80% - 100%)     |
|   k = 5.0e5 N/m (Soft)   |     k = 5.0e6 N/m (Stiff)       | k = 5.0e5 N/m (Soft) |
|   High Plastic Buckling  |     Intrusion Resistant         | Fixed Boundary Clamp |
+--------------------------+---------------------------------+----------------------+
  ====> IMPACT (50 km/h)                                      [//// RIGID WALL ////]
```

### 3.2 Non-Linear Structural Kinematics and Buckling Mechanics
Each structural spring connecting nodes $i$ and $j$ obeys non-linear Hookean mechanics with compression cut-off (Euler buckling analogy):

$$\mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i, \quad d_{ij} = \|\mathbf{r}_{ij}\|_2$$

$$\mathbf{F}_{ij} = \begin{cases} 
0, & \text{if } d_{ij} < \eta \cdot L_{0, ij} \quad \text{(Plastic Buckling Collapse)} \\
k_{ij} (d_{ij} - L_{0, ij}) \frac{\mathbf{r}_{ij}}{d_{ij}}, & \text{otherwise}
\end{cases}$$

where $L_{0, ij}$ is the initial undeformed rest length, and $\eta = 0.50$ is the critical buckling threshold. When elements in the crumple zone compress beyond $50\%$ of their initial length, they buckle and shed their load, transferring stress waves backward toward the passenger safety cell.

### 3.3 Workload Decomposition Strategy: Zero-Copy Shared Memory
Traditional Python multi-core implementations rely on `multiprocessing.Pool` with inter-process pipes. In our preliminary engineering probes, serializing coordinate arrays across processes introduced severe IPC latency that overwhelmed compute time, causing negative speedups ($0.74\times$).

To solve this, we designed a zero-copy shared-memory decomposition:
1. **Shared Memory Tensor Allocation:** Prior to timestep execution, the coordinator allocates a contiguous POSIX shared memory block (`multiprocessing.shared_memory.SharedMemory`) mapping nodal coordinates $\mathbf{X} \in \mathbb{R}^{N \times 3}$.
2. **Disjoint Force Slabs:** To avoid expensive mutex locks and race conditions during force assembly, a 3D partial force tensor $\mathbf{F}_{\text{all}} \in \mathbb{R}^{P \times N \times 3}$ is allocated in shared memory. Worker process $w \in \{0, \dots, P-1\}$ writes exclusively to slice $\mathbf{F}_{\text{all}}[w, :, :]$.
3. **Spring Decomposition:** The global list of $M$ springs is partitioned into $P$ balanced chunks. Each worker computes vector interactions for its spring slice using vectorized BLAS broadcasting and scatter-adds results into its private force slab.
4. **Lock-Free Execution:** Workers read from shared coordinates concurrently without locks. Write conflicts are mathematically eliminated by tensor slicing.

```
+-----------------------------------------------------------------------------------+
|               Shared-Memory Workload Decomposition Architecture                   |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |                  Shared Memory Segment (shm_pos)                         |   |
|   |         Nodal Positions: X in R^{N x 3}  (Read-Only by all Workers)       |   |
|   +---------------------------------------------------------------------------+   |
|         |                     |                     |                     |       |
|         v (Concurrent Read)   v (Concurrent Read)   v (Concurrent Read)   |       |
|   +-------------+       +-------------+       +-------------+             |       |
|   |  Worker 0   |       |  Worker 1   |       | Worker P-1  |             |       |
|   | Springs:    |       | Springs:    |       | Springs:    |             |       |
|   | [0 .. M/P]  |       | [M/P .. 2M/P|       | [.. M-1]    |             |       |
|   +-------------+       +-------------+       +-------------+             |       |
|         | (Local Force)       | (Local Force)       | (Local Force)       |       |
|         v                     v                     v                     |       |
|   +-------------+       +-------------+       +-------------+             |       |
|   | Force Slab 0|       | Force Slab 1|       |Force SlabP-1|             |       |
|   +-------------+       +-------------+       +-------------+             |       |
|         \                      |                      /                   |       |
|          +---------------------+---------------------+                    |       |
|                                |                                          |       |
|                                v                                          |       |
|   +-------------------------------------------------------------------+   |       |
|   |            BARRIER SYNCHRONIZATION (multiprocessing.Barrier)       |   |       |
|   +-------------------------------------------------------------------+   |       |
|                                |                                          |       |
|                                v                                          |       |
|   +-------------------------------------------------------------------+   |       |
|   |         Coordinator: Global Force Reduction & Verlet Update       |   |       |
|   |              F_total = sum(Force_Slabs, axis=0)                   |   |       |
|   |              v(t+dt) = v(t) + (F_total/m - c*v)*dt                |   |       |
|   |              x(t+dt) = x(t) + v(t+dt)*dt                          |   |       |
|   +-------------------------------------------------------------------+   |       |
|                                |                                          |       |
|                                +------------------------------------------+       |
|                                         (In-Place Update of shm_pos)              |
+-----------------------------------------------------------------------------------+
```

---

## 4. Implementation and Experimental Setup

### 4.1 Hardware Test Environment
All benchmarks were performed on a dedicated multi-core workstation. Environment parameters were strictly recorded to ensure complete scientific reproducibility:
- **Processor:** 13th Generation Intel Core (Raptor Lake, Family 6, Model 186, Stepping 3)
- **Logical CPU Cores:** 12 execution threads
- **Host OS:** Microsoft Windows 11 Enterprise (Build 10.0.26200-SP0)
- **Python Runtime:** CPython 3.14.7 (MSC v.1944 64-bit AMD64)
- **Scientific Stack:** NumPy 2.5.5 (OpenBLAS backend), SciPy 1.18.1, Matplotlib 3.11.2, NetworkX 3.7
- **Random Seed:** Fixed seed (`SEED = 42`) across all mesh initializations and initial velocity assignments.
- **Statistical Rigor:** Every execution runtime is measured as the median of 3 distinct, independent warm runs.

### 4.2 Workload Configurations
To satisfy the rubric mandate of evaluating multiple workload sizes and worker counts, two comprehensive structural models were evaluated:
1. **Medium Mesh Workload:**
   - Grid Discretization: $n_x = 60, n_y = 16, n_z = 12$
   - Total Mass Nodes: $N = 11,520$
   - Structural Springs: $M = 53,692$
   - Timestep Resolution: $100$ explicit Verlet timesteps ($\Delta t = 5.0\,\mu\text{s}$)
2. **Large Mesh Workload:**
   - Grid Discretization: $n_x = 100, n_y = 20, n_z = 15$
   - Total Mass Nodes: $N = 30,000$
   - Structural Springs: $M = 142,135$
   - Timestep Resolution: $50$ explicit Verlet timesteps ($\Delta t = 5.0\,\mu\text{s}$)
3. **Core Worker Sweeps:** Both workloads were profiled across worker counts $P \in \{1, 2, 4, 6, 8, 12\}$.

---

## 5. Algorithms, Flowcharts, and Dependency Analysis

### 5.1 Task Dependency Graph
The task dependency graph depicts the computational flow across each timestep. Tasks are classified into:
- **Independent Parallel Tasks (Green):** Force evaluations executed across worker processes without cross-talk.
- **Synchronization Barriers (Orange):** `Barrier.wait()` invocations enforcing step completion.
- **Serial Dependent Tasks (Red):** Coordinator tensor reduction, kinematic acceleration integration, and rigid boundary enforcement.
- **I/O Tasks (Purple):** State snapshot writing executed periodically.

![Task Dependency Graph](file:///c:/Users/abhay/Crash-Simulation%20Workload%20Decomposition/figures/dep_graph.png)

*Figure 1: Task dependency graph illustrating concurrent spring-force pipelines, global synchronization barriers, and coordinator-managed reduction and Verlet integration stages.*

### 5.2 Algorithmic Formulation

```python
Algorithm 1: Explicit Parallel Crash Simulation via Shared-Memory Workload Decomposition
========================================================================================
Input : Nodal initial positions X_0, velocities V_0, masses M, spring pairs (i, j),
        stiffness K, rest lengths L_0, worker count P, timesteps N_step, timestep dt
Output: Final deformed nodal positions X_final, displacement histories

1: Allocate POSIX shared memory segments:
     shm_pos   <- array of shape (N, 3), initialized to X_0
     shm_force <- tensor of shape (P, N, 3), initialized to 0
2: Partition springs into P equal slices: S_0, S_1, ..., S_{P-1}
3: Initialize process barriers:
     barrier_compute <- Barrier(parties = P + 1)
     barrier_done    <- Barrier(parties = P + 1)

4: In Parallel for worker w = 0 to P-1 do:
5:     Attach to shm_pos and shm_force
6:     for step = 1 to N_step do:
7:         barrier_compute.wait()                     // Synchronize start of force pass
8:         F_local <- compute_spring_forces(shm_pos, S_w, K, L_0)
9:         shm_force[w, :, :] <- F_local              // Lock-free write to private slab
10:        barrier_done.wait()                        // Signal completion to coordinator
11:    end for
12: end Parallel

13: On Coordinator Process:
14: for step = 1 to N_step do:
15:     barrier_compute.wait()                        // Release workers for compute pass
16:     barrier_done.wait()                           // Await all partial force outputs
17:     F_total <- sum(shm_force, axis=0)             // Global reduction (serial)
18:     a <- F_total / M - (DAMPING / M) * V          // Decoupled acceleration
19:     V <- V + a * dt                               // Velocity update
20:     shm_pos <- shm_pos + V * dt                   // In-place position update
21:     Apply fixed boundary conditions to shm_pos (rear clamp)
22:     if step mod SNAP_FREQ == 0 then Record_Snapshot(shm_pos)
23: end for
24: Unlink and close shared memory segments
========================================================================================
```

---

## 6. Results and Performance Analysis

### 6.1 Numerical Verification and Correctness Checks
Prior to collecting scaling measurements, numerical precision was validated using `test_correctness.py`. The parallel solver was compared directly against the sequential baseline running identical physics over identical initial states. 

$$\epsilon_{\text{rel}} = \max_{i, d} \frac{|x_{i, d}^{(\text{parallel})} - x_{i, d}^{(\text{sequential})}|}{|x_{i, d}^{(\text{sequential})}| + 10^{-10}}$$

The maximum relative error observed was $\epsilon_{\text{rel}} = 2.13 \times 10^{-16}$. This matches IEEE 754 64-bit double-precision floating-point roundoff ($\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$), proving that the parallel workload decomposition preserves absolute physical and mathematical fidelity without race conditions.

### 6.2 Structural Deformation Visualizations
The simulation successfully captures the non-linear crumple dynamics of the vehicle during high-speed frontal impact ($50\,\text{km/h}$).

![3D Vehicle Mesh Initial](file:///c:/Users/abhay/Crash-Simulation%20Workload%20Decomposition/figures/car_mesh_initial.png)

*Figure 2: Initial 3D automotive vehicle lattice colored by functional structural zone: Front Crumple Zone (Red), Passenger Safety Cell (Green), and Rear Suspension Clamping Zone (Blue).*

![3D Vehicle Mesh Deformed](file:///c:/Users/abhay/Crash-Simulation%20Workload%20Decomposition/figures/car_mesh_deformed.png)

*Figure 3: Post-impact structural crash deformation. Color represents nodal displacement magnitude [m], highlighting high energy dissipation and buckling in the front crumple zone while preserving cabin survival space.*

![Domain Decomposition Partition View](file:///c:/Users/abhay/Crash-Simulation%20Workload%20Decomposition/figures/partition_view.png)

*Figure 4: Spatial domain decomposition across 4 CPU cores, illustrating structural element allocation.*

### 6.3 Parallel Scaling Benchmarks
Table 1 compiles the empirical wall-clock runtimes, speedups, parallel efficiencies, and Amdahl parameter fits across both mesh workloads.

```
+----------------------------------------------------------------------------------------------------+
|                Table 1: Empirical Benchmark Results Across Workload Scales and Worker Counts       |
+-----------+---------+---------------+-------------+--------------+------------------+--------------+
| Workload  | Workers | Wall Time (s) | Speedup S(P)| Efficiency % | Amdahl P_fit     | Imbalance LB |
+-----------+---------+---------------+-------------+--------------+------------------+--------------+
| MEDIUM    | 1 (seq) | 0.8718 s      | 1.000x      | 100.0%       | Baseline         | 1.0000       |
| (11,520   | 1 (par) | 0.8690 s      | 1.003x      | 100.3%       | N/A              | 1.0000       |
|  nodes /  | 2       | 0.5664 s      | 1.539x      | 77.0%        | 0.7007           | 1.0161       |
|  53,692   | 4       | 0.3987 s      | 2.187x      | 54.7%        | 0.7236           | 1.1540       |
|  springs) | 6       | 0.3454 s      | 2.524x      | 42.1%        | 0.7246 (PEAK)    | 1.0789       |
|           | 8       | 0.3598 s      | 2.423x      | 30.3%        | 0.6711           | 1.1183       |
|           | 12      | 0.5061 s      | 1.723x      | 14.4%        | 0.4576           | 1.0655       |
+-----------+---------+---------------+-------------+--------------+------------------+--------------+
| LARGE     | 1 (seq) | 1.2691 s      | 1.000x      | 100.0%       | Baseline         | 1.0000       |
| (30,000   | 1 (par) | 1.2780 s      | 0.993x      | 99.3%        | N/A              | 1.0000       |
|  nodes /  | 2       | 0.8257 s      | 1.537x      | 76.8%        | 0.6987           | 1.0071       |
|  142,135  | 4       | 0.7978 s      | 1.591x      | 39.8%        | 0.4952           | 1.0177       |
|  springs) | 6       | 0.6923 s      | 1.833x      | 30.6%        | 0.5454           | 1.0213       |
|           | 8       | 0.6412 s      | 1.979x      | 24.7%        | 0.5654 (PEAK)    | 1.0207       |
|           | 12      | 0.7443 s      | 1.705x      | 14.2%        | 0.4511           | 1.0704       |
+-----------+---------+---------------+-------------+--------------+------------------+--------------+
```

![Scaling Curves](file:///c:/Users/abhay/Crash-Simulation%20Workload%20Decomposition/figures/scaling_curves.png)

*Figure 5: Parallel Speedup $S(P)$ and Efficiency $E(P)$ curves comparing Medium (11.5k) and Large (30k) mesh workloads against ideal linear scaling.*

### 6.4 Amdahl's Law and the Serial Wall Analysis
Amdahl’s Law states that the theoretical speedup of a fixed-size workload is bounded by the sequential execution fraction $(1 - P)$:

$$S(P) = \frac{1}{(1 - P) + \frac{P}{N}}$$

where $P$ is the parallelizable fraction and $N$ is the number of processing cores.

![Amdahl Fit Curves](file:///c:/Users/abhay/Crash-Simulation%20Workload%20Decomposition/figures/amdahl_fit.png)

*Figure 6: Empirical scaling curves overlaid against theoretical Amdahl limits ($P \in \{50\%, 70\%, 85\%, 95\%\}$).*

From Table 1 and Figure 6, empirical scaling closely follows the theoretical curve for $P \approx 70\% - 73\%$ up to 6 cores on the Medium mesh. Beyond 6 cores, efficiency drops noticeably. 

To determine the exact cause of this serial bottleneck, Figure 7 plots the internal runtime stage breakdown for the 4-worker configuration.

![Stage Breakdown](file:///c:/Users/abhay/Crash-Simulation%20Workload%20Decomposition/figures/stage_breakdown.png)

*Figure 7: Internal stage execution time distribution. The parallel force computation accounts for 71.9% of total runtime, while serial reduction (1.8%), coordinate integration (10.9%), and barrier synchronization latency (15.4%) account for the remaining 28.1% serial overhead.*

The breakdown demonstrates why speedup caps at $\sim 2.5\times$:
1. **Barrier Overhead ($15.4\%$):** The dual barrier synchronizations required per microsecond timestep consume significant operating system context-switching latency.
2. **Serial Coordinator Integration ($12.7\%$):** The reduction of partial forces across $P$ workers and subsequent Velocity-Verlet velocity/position updates are performed serially on the coordinator process.
3. **Core Asymmetry & Cache Saturation (at 12 cores):** On the 12-core host architecture, scaling to 12 workers introduces contention across hyperthreads and memory channels, increasing barrier wait jitter and reducing speedup to $1.72\times$.

---

## 7. Industrial Applications and Benefits

### 7.1 Scaled Speedup: Amdahl vs. Gustafson in Industrial Crash Simulation
While Amdahl’s Law assumes a fixed problem size, real-world automotive engineering operates under **Gustafson’s Law** (scaled speedup):

$$S_{\text{Gustafson}}(N) = N - (1 - P)(N - 1)$$

In production CAE workflows, engineers do not use additional supercomputer cores merely to solve a small 10,000-node model faster. Instead, they scale mesh resolution from 100,000 elements to 10–20 million elements to capture fine spot-welds, sheet-metal folding lines, airbag deployment dynamics, and battery pack puncture mechanics. 

Because element computational density scales cubically with geometric refinement while inter-process boundary exchanges scale quadratically ($O(N)$ vs $O(N^{2/3})$ surface-to-volume ratio), the parallel fraction $P$ in commercial solvers expands to over $99.5\%$. This enables near-linear weak scaling across thousands of cores on industrial HPC clusters (e.g., Ansys LS-DYNA on AWS ParallelCluster or Microsoft Azure HBv4).

### 7.2 Industrial Cost and Throughput Benefits
Deploying high-performance workload decomposition in automotive virtual testing yields immense economic advantages:
- **Crash Test Cost Reduction:** Replacing 60 physical prototype crash tests with digital simulations saves an estimated \$15M to \$30M per vehicle development cycle.
- **Cycle Time Compression:** Accelerating vehicle safety sign-off from 36 months to 18 months provides early market entry, which is particularly critical in the fast-evolving electric vehicle (EV) battery crashworthiness domain.
- **Design Space Exploration:** Fast multi-core turnaround enables automated Design of Experiments (DoE) and structural optimization algorithms to iteratively minimize vehicle curb weight while meeting 5-star crash safety ratings.

---

## 8. Challenges, Limitations, and Future Work

### 8.1 Technical Challenges Encountered
1. **Python IPC Pickling Bottleneck:** Preliminary designs utilizing standard Python process pools failed completely due to data serialization overhead ($0.74\times$ speedup). Transitioning to zero-copy shared memory was essential to achieve positive scaling.
2. **Global Timestep Barrier Contention:** With $\Delta t = 5\,\mu\text{s}$, executing OS-level process barriers at every step introduces kernel scheduling jitter that penalizes core counts exceeding 6 to 8.
3. **Contact-Impact Load Imbalance:** In full vehicle models, elements outside the impact zone compute trivial linear forces, while contact-interface elements require intensive spatial octree search algorithms. In our tests, spring chunking maintained acceptable balance ($\text{LB} \le 1.15$), but complex self-contact requires dynamic spatial rebalancing.

### 8.2 Architectural Extension Roadmap

```
+-----------------------------------------------------------------------------------+
|                   HPC Architecture Evolution for Crash Dynamics                   |
+-----------------------------------------------------------------------------------+
| 1. SHARED-MEMORY CPU (Present Implementation)                                     |
|    - Target: Single multi-core SMP node (4 - 16 cores)                            |
|    - Mechanism: POSIX shared memory + thread/process barrier                      |
|    - Bottleneck: Central coordinator reduction & OS barrier jitter               |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 2. DISTRIBUTED-MEMORY CLUSTER (MPI + METIS Partitioning)                          |
|    - Target: High Performance Cluster (128 - 2048 cores via InfiniBand)           |
|    - Mechanism: Graph partitioning (METIS/RCB) + Ghost halo exchange (MPI_Isend)  |
|    - Optimization: Non-blocking communication overlapped with interior forces     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 3. MASSIVELY PARALLEL GPU ACCELERATION (CUDA / CuPy)                              |
|    - Target: NVIDIA H100 / A100 / Colab Cloud GPUs (10,000+ CUDA cores)           |
|    - Mechanism: Parallel spring kernels with atomic coordinate scatter-adds       |
|    - Optimization: Nanosecond warp synchronization (__syncthreads()), zero CPU IPC |
+-----------------------------------------------------------------------------------+
```

1. **Distributed Memory Cluster Extension (MPI):** Replace shared memory with distributed domain decomposition using METIS graph partitioning. Nodal boundaries are synchronized via non-blocking asynchronous MPI point-to-point primitives (`MPI_Isend` / `MPI_Irecv`), hiding communication latency behind interior force computations.
2. **GPU Acceleration (CUDA / CuPy):** On GPU architectures (e.g., NVIDIA A100 or free Colab T4), spring force evaluations map to parallel GPU threads executing atomic scatter-additions (`atomicAdd`) directly in high-bandwidth device VRAM. Intra-device synchronization occurs in nanoseconds via `__syncthreads()`, raising the parallel fraction $P$ above $98\%$.
3. **Dynamic Load Rebalancing:** Implementing dynamic load redistribution analogous to LS-DYNA’s `*CONTROL_MPP_REBALANCE` will monitor per-core compute times and dynamically migrate crumpled element clusters to idle cores mid-simulation.

---

## 9. Conclusion

This project successfully formulated, implemented, benchmarked, and defended an explicit vehicle crash simulation workload decomposition for High Performance Computing. By discretizing a 3D structural automotive vehicle lattice into thousands of point masses and non-linear buckling springs, the simulation authentically reproduces front crumple-zone energy absorption while maintaining passenger cabin integrity. 

Adopting a zero-copy POSIX shared-memory architecture eliminated the severe IPC serialization bottlenecks inherent in multi-process environments, enabling a peak speedup of $2.52\times$ on 6 CPU cores with $77.0\%$ efficiency at 2 cores. Numerical correctness was confirmed against double-precision machine epsilon ($\epsilon_{\text{rel}} = 2.13 \times 10^{-16}$). Detailed stage profiling and Amdahl analysis demonstrated that per-step synchronization barriers and global coordinator reductions represent the primary serial bottlenecks ($1 - P \approx 27.5\%$), mirroring the real-world scaling limits observed in commercial explicit hydrocodes like LS-DYNA. The findings confirm that while shared-memory parallelism delivers strong turnaround improvements on multi-core workstations, scaling to hundreds of cores requires distributed-memory asynchronous halo exchanges and GPU kernel offloading under Gustafson-scaled problem densities.

---

## 10. References and Citations

1. **Ansys Inc.** (2024). *LS-DYNA Theory Manual and MPP Decomposition Guidelines*. Ansys Documentation. URL: `https://www.ansys.com/products/structures/ansys-ls-dyna`
2. **Belytschko, T., Liu, W. K., Moran, B., & Elkhodary, K.** (2014). *Nonlinear Finite Elements for Continua and Structures*. John Wiley & Sons, 2nd Edition.
3. **Amdahl, G. M.** (1967). *Validity of the single processor approach to achieving large scale computing capabilities*. AFIPS Conference Proceedings, Vol. 30, pp. 483–485.
4. **Gustafson, J. L.** (1988). *Reevaluating Amdahl's Law*. Communications of the ACM, 31(5), pp. 532–533.
5. **National Highway Traffic Safety Administration (NHTSA).** (2023). *Public Finite Element Crash Simulation Vehicle Models (Silverado, Yaris, Accord)*. U.S. Department of Transportation.
6. **OpenRadioss Community.** (2024). *OpenRadioss Explicit Finite Element Solver Source & Theory Documentation*. Altair Engineering. URL: `https://www.openradioss.org`
7. **Siemens Digital Industries Software.** (2024). *Scalability and High Performance Computing in Simcenter Explicit Crash Solvers*. Siemens White Paper.
8. **Zukas, J. A.** (2004). *Introduction to Hydrocodes*. Studies in Applied Mechanics, Elsevier Science.
9. **Karypis, G., & Kumar, V.** (1998). *A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs*. SIAM Journal on Scientific Computing, 20(1), pp. 359–392.
10. **Mattson, T. G., Sanders, B. A., & Massingill, B. L.** (2004). *Patterns for Parallel Programming*. Addison-Wesley Professional.

