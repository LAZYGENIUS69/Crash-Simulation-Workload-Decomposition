# Verified References and Asset Attributions

Course: High Performance Computing (24ISE503) — Activity 01  
Repository: `LAZYGENIUS69/Crash-Simulation-Workload-Decomposition`

---

## 1. Academic & Industrial References (Peer-Reviewed / DOI-Verified)

1. **Reichert, R., Mohan, P., Marzougui, D., Kan, C.-D., & Brown, D. (2016).**  
   *Validation of a Detailed Toyota Camry Finite Element Model for Multiple Impact Configurations.*  
   SAE Technical Paper 2016-01-1534.  
   DOI: [10.4271/2016-01-1534](https://doi.org/10.4271/2016-01-1534)  
   *(Provides empirical vehicle crash deceleration profiles, barrier intrusion metrics, and front-structure load distribution curves).*

2. **Karp, A. H., & Flatt, H. P. (1990).**  
   *Measuring parallel processor performance.*  
   Communications of the ACM, 33(5), 539–543.  
   DOI: [10.1145/78607.78614](https://doi.org/10.1145/78607.78614)  
   *(Theoretical derivation of the Karp-Flatt metric $e$ for detecting serial bottlenecks and communication overheads in parallel scaling).*

3. **Amdahl, G. M. (1967).**  
   *Validity of the single processor approach to achieving large scale computing capabilities.*  
   Proceedings of the AFIPS Spring Joint Computer Conference, 483–485.  
   DOI: [10.1145/1465482.1465560](https://doi.org/10.1145/1465482.1465560)  
   *(Foundational fixed-workload speedup law).*

4. **Gustafson, J. L. (1988).**  
   *Reevaluating Amdahl's law.*  
   Communications of the ACM, 31(5), 532–533.  
   DOI: [10.1145/42411.42415](https://doi.org/10.1145/42411.42415)  
   *(Scaled-workload speedup formulation applied to increasing finite-element mesh resolution).*

5. **Karypis, G., & Kumar, V. (1998).**  
   *A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs.*  
   SIAM Journal on Scientific Computing, 20(1), 359–392.  
   DOI: [10.1137/S1064827595287997](https://doi.org/10.1137/S1064827595287997)  
   *(Algorithmic foundation for graph partitioning in FEM/structural mechanics domain decomposition).*

6. **McCalpin, J. D. (1995).**  
   *Memory Bandwidth and Machine Balance in Current High Performance Computers.*  
   IEEE Computer Society Technical Committee on Computer Architecture (TCCA) Newsletter, 19–25.  
   *(Standard STREAM benchmark methodology for determining sustained memory bandwidth limits).*

7. **Verlet, L. (1967).**  
   *Computer "Experiments" on Classical Fluids. I. Thermodynamical Properties of Lennard-Jones Molecules.*  
   Physical Review, 159(1), 98–103.  
   DOI: [10.1103/PhysRev.159.98](https://doi.org/10.1103/PhysRev.159.98)  
   *(Symplectic time-integration formulation ensuring long-term energy conservation in particle and spring-mass mechanics).*

---

## 2. Software Tools & Open Source Licenses

- **GNU Compiler Collection (GCC 13.3.0):** GNU GPLv3 with Runtime Library Exception.
- **OpenMP API (v4.5 / v5.0):** OpenMP Architecture Review Board.
- **Python Software Foundation (Python 3.12 / 3.14):** PSF License.
- **NumPy, SciPy, Pandas:** BSD 3-Clause License.
- **Matplotlib:** PSF-based Matplotlib License.
- **NetworkX:** BSD 3-Clause License.

---

## 3. Geometry & Asset Provenance

- **Mesh Models:** All 3D vehicle geometries, spring-mass lattices, boundary masks, and crumple-zone definitions in this repository are procedurally generated in software (`src/car_mesh.py`). No proprietary CAD or LS-DYNA mesh files are used.
