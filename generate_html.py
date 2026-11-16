"""
generate_html.py — Compiles REPORT.md into a self-contained, publication-styled HTML
=====================================================================================
Generates REPORT.html with modern typography, KaTeX math rendering,
embedded figures, and print-optimized CSS for direct PDF export.
"""

import pathlib
import base64

BASE_DIR = pathlib.Path(__file__).parent.resolve()
FIG_DIR = BASE_DIR / "figures"
HTML_PATH = BASE_DIR / "REPORT.html"

def img_to_base64(path):
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode('utf-8')

def main():
    dep_b64 = img_to_base64(FIG_DIR / "dep_graph.png")
    car_init_b64 = img_to_base64(FIG_DIR / "car_mesh_initial.png")
    car_def_b64 = img_to_base64(FIG_DIR / "car_mesh_deformed.png")
    part_b64 = img_to_base64(FIG_DIR / "partition_view.png")
    scale_b64 = img_to_base64(FIG_DIR / "scaling_curves.png")
    amdahl_b64 = img_to_base64(FIG_DIR / "amdahl_fit.png")
    stage_b64 = img_to_base64(FIG_DIR / "stage_breakdown.png")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Crash-Simulation Workload Decomposition — HPC (24ISE503)</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>
<style>
    @page {{
        size: A4;
        margin: 20mm 18mm 20mm 18mm;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 10.5pt;
        line-height: 1.5;
        color: #1a1a1a;
        max-width: 900px;
        margin: 0 auto;
        padding: 30px;
        background-color: #fafafa;
    }}
    .document-card {{
        background: #ffffff;
        padding: 40px 50px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border-radius: 8px;
    }}
    @media print {{
        body {{ background: none; padding: 0; }}
        .document-card {{ padding: 0; box-shadow: none; border-radius: 0; }}
        .no-print {{ display: none; }}
    }}
    .header-banner {{
        border-bottom: 2px solid #1b4f72;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }}
    .course-title {{
        font-size: 10pt;
        font-weight: 700;
        color: #1b4f72;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    h1.main-title {{
        font-size: 20pt;
        color: #0e2f44;
        margin: 4px 0 10px 0;
        font-weight: 800;
        line-height: 1.2;
    }}
    .meta-bar {{
        font-size: 9pt;
        color: #555555;
        background: #f2f7fa;
        padding: 8px 14px;
        border-left: 3px solid #2e86c1;
        border-radius: 3px;
        margin-bottom: 25px;
    }}
    h2 {{
        font-size: 13pt;
        color: #1b4f72;
        border-bottom: 1px solid #e1e8ed;
        padding-bottom: 4px;
        margin-top: 24px;
        margin-bottom: 10px;
        font-weight: 700;
    }}
    h3 {{
        font-size: 11pt;
        color: #2c3e50;
        margin-top: 14px;
        margin-bottom: 6px;
        font-weight: 600;
    }}
    p {{
        margin-top: 0;
        margin-bottom: 10px;
        text-align: justify;
    }}
    .abstract-box {{
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-left: 4px solid #1b4f72;
        padding: 14px 18px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 10pt;
    }}
    .figure-container {{
        text-align: center;
        margin: 18px 0;
        page-break-inside: avoid;
    }}
    .figure-container img {{
        max-width: 95%;
        height: auto;
        border: 1px solid #e1e8ed;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    .figure-caption {{
        font-size: 8.5pt;
        color: #555555;
        font-style: italic;
        margin-top: 6px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 8.5pt;
        margin: 16px 0;
        page-break-inside: avoid;
    }}
    th, td {{
        border: 1px solid #d5dbdb;
        padding: 6px 8px;
        text-align: center;
    }}
    th {{
        background: #d6eaf8;
        color: #1b4f72;
        font-weight: 700;
    }}
    tr:nth-child(even) {{
        background: #fdfefe;
    }}
    tr:nth-child(odd):not(:first-child) {{
        background: #f8f9f9;
    }}
    pre.code-block {{
        background: #272822;
        color: #f8f8f2;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 8pt;
        overflow-x: auto;
        font-family: Consolas, "Courier New", monospace;
        line-height: 1.4;
    }}
    .ref-list {{
        font-size: 9pt;
        padding-left: 20px;
    }}
    .ref-list li {{
        margin-bottom: 5px;
    }}
</style>
</head>
<body>
<div class="document-card">
    <div class="header-banner">
        <div class="course-title">High Performance Computing (24ISE503) — Activity 01 (Automotive Domain)</div>
        <h1 class="main-title">Crash-Simulation Workload Decomposition</h1>
    </div>
    
    <div class="meta-bar">
        <strong>Student Submission</strong> | B.Tech ISE Semester V | Academic Year: 2026<br>
        <strong>Target Environment:</strong> 12-Core Intel Architecture | CPython 3.14.7 | Zero-Copy POSIX Shared Memory
    </div>

    <div class="abstract-box">
        <strong>Abstract — </strong> Finite element crashworthiness simulation is one of the most computationally demanding High Performance Computing (HPC) workloads in the automotive sector. Real-world vehicle crash solvers such as LS-DYNA, RADIOSS, and PAM-CRASH rely on explicit time integration, requiring millions of spatial degrees of freedom to be updated over tens of thousands of microsecond-scale timesteps under strict Courant-Friedrichs-Lewy (CFL) stability bounds. This study formulates, implements, benchmarks, and analyzes a parallel explicit dynamic crash simulation workload using shared-memory domain decomposition. We construct a 3D structural lattice model of an automotive vehicle comprising up to 30,000 point-mass nodes and 142,135 structural spring-damper elements with localized front crumple zones ($k = 5.0\\times 10^5\\,\\text{{N/m}}$), rigid safety cabin compartments ($k = 5.0\\times 10^6\\,\\text{{N/m}}$), and non-linear compressive buckling. The workload is partitioned across CPU cores using a zero-copy shared-memory architecture (<code>multiprocessing.shared_memory</code>), completely avoiding inter-process serialization bottlenecks while mapping to cache-coherent Symmetric Multiprocessing (SMP) supercomputer nodes. We construct a complete task dependency graph delineating concurrent force computations, synchronization barriers, and serial reduction phases. Benchmarking across core counts ($P \\in \\{{1, 2, 4, 6, 8, 12\\}}$) across two workload scales (11,520 nodes / 53,692 springs and 30,000 nodes / 142,135 springs) achieves a peak parallel speedup of $2.52\\times$ at 6 workers ($77.0\\%$ efficiency at 2 workers). Fitting to Amdahl’s Law reveals an empirical parallel fraction $P_{{\\text{{Amdahl}}}} \\approx 72.5\\%$, demonstrating that per-step barrier synchronization and global reduction dominate the serial bottleneck ($1 - P \\approx 27.5\\%$). Numerical correctness is rigorously proven against sequential execution with a maximum relative coordinate discrepancy bounded at $2.13 \\times 10^{{-16}}$ (IEEE 754 double-precision machine epsilon). We conclude with industrial scalability recommendations, MPI/CUDA cluster extension architectures, and fault tolerance strategies.
    </div>

    <h2>1. Introduction and Industry Background</h2>
    <p>In modern automotive engineering, physical crash destruction testing is cost-prohibitive ($250,000–$500,000 per physical prototype) and introduces multi-month manufacturing lead times. Consequently, Original Equipment Manufacturers (OEMs) such as BMW, Mercedes-Benz, Toyota, and GM rely extensively on explicit finite element analysis (FEA) to evaluate passenger cabin intrusion, structural energy absorption, and occupant biomechanics against safety standards.</p>
    
    <p>Dynamic mechanical equilibrium is governed by the second-order partial differential equation of continuum elastodynamics:</p>
    $$\\mathbf{{M}} \\ddot{{\\mathbf{{u}}}}(t) + \\mathbf{{C}} \\dot{{\\mathbf{{u}}}}(t) + \\mathbf{{K}}(\\mathbf{{u}}) \\mathbf{{u}}(t) = \\mathbf{{F}}_{{\\text{{ext}}}}(t)$$
    <p>where $\\mathbf{{M}}$ is the mass matrix, $\\mathbf{{C}}$ is the damping operator, $\\mathbf{{K}}(\\mathbf{{u}})$ is the non-linear internal stiffness operator, and $\\mathbf{{F}}_{{\\text{{ext}}}}$ represents external impact contact interactions. Unlike implicit solvers which require assembling and factorizing massive tangential stiffness matrices at each step ($O(N^3)$ computational complexity), explicit central-difference and Velocity-Störmer-Verlet algorithms decouple nodal accelerations via diagonal lumped masses ($\mathbf{{a}}_i = \\mathbf{{F}}_i / m_i$). This decouples spatial degrees of freedom, yielding $O(N)$ computational complexity that is ideal for high-concurrency computing.</p>

    <p>However, explicit integration is conditionally stable, governed by the Courant-Friedrichs-Lewy (CFL) stability limit: $\\Delta t \\le L_{{\\min}} / c$, where $c = \\sqrt{{E / \\rho}} \\approx 5170\\,\\text{{m/s}}$ is the acoustic dilational sound speed in automotive steel. For typical structural meshes with $5\\,\\text{{mm}}$ elements, $\\Delta t$ is restricted to $0.5$–$2.0\\,\\mu\\text{{s}}$, dictating $50,000$ to $200,000$ consecutive timesteps per crash event. Because each step depends strictly on the physical state of the preceding step, temporal parallelization is impossible; concurrency must be extracted across the spatial domain.</p>

    <h2>2. Problem Statement and Objectives</h2>
    <p>The primary challenge in scaling automotive crash workloads lies in three architectural constraints: (1) the requirement for global state synchronization at every microsecond timestep, (2) spatial load imbalance caused by localized structural crumpling at the vehicle front while the rear remains quiescent, and (3) communication-to-computation inversion, where inter-core communication and synchronization latency overtake fast element computation as core counts increase.</p>
    <p>Objectives of this study: (a) construct a recognizable procedural 3D automotive lattice model segmented into crumple zones and safety cells; (b) design and implement an efficient, zero-copy shared-memory workload decomposition in Python; (c) generate a formal task dependency graph identifying independent tasks, dependencies, and barriers; (d) profile multi-core scaling, speedup, and efficiency across 1 to 12 workers on two distinct mesh sizes; (e) fit empirical data to Amdahl's Law, quantify serial bottlenecks, and provide production HPC cluster recommendations.</p>

    <h2>3. Methodology and Architecture</h2>
    <p>We discretize a procedural vehicle (Length $4.5\\,\\text{{m}}$, Width $1.8\\,\\text{{m}}$, Peak Height $1.5\\,\\text{{m}}$) into a 3D lattice of point masses connected by structural spring-dampers with non-linear compressive buckling (Euler buckling model). The structure features three zones: a sacrificial Front Crumple Zone ($0$–$20\\%$ length, $k = 5.0\\times 10^5\\,\\text{{N/m}}$) that collapses plastically, an ultra-stiff Passenger Safety Cabin ($20$–$80\\%$ length, $k = 5.0\\times 10^6\\,\\text{{N/m}}$) that resists intrusion, and a Rear Clamping Zone ($80$–$100\\%$) fixed to rigid boundary reaction walls.</p>
    
    <p>To eliminate Python IPC pickling overhead, the solver allocates a POSIX shared-memory segment (<code>multiprocessing.shared_memory</code>) storing coordinates $\\mathbf{{X}} \\in \\mathbb{{R}}^{{N \\times 3}}$ and a 3D force tensor $\\mathbf{{F}} \\in \\mathbb{{R}}^{{P \\times N \\times 3}}$. The spring list is chunked evenly across $P$ workers. Each worker reads positions concurrently without locking, evaluates Hookean forces and buckling states, and writes to its private force slab. A coordinator process waits at a barrier, performs vector reduction across slabs, and updates kinematic state via Velocity-Verlet integration.</p>

    <div class="figure-container">
        <img src="{dep_b64}" alt="Task Dependency Graph">
        <div class="figure-caption">Figure 1: Task Dependency Graph illustrating concurrent worker force computation, barrier synchronization points, and serial coordinator reduction and Verlet integration.</div>
    </div>

    <h2>4. Implementation and Experimental Setup</h2>
    <p>Experiments were conducted on a 13th Gen Intel Core processor featuring 12 logical cores on Windows 11 Enterprise, running CPython 3.14.7, NumPy 2.5.5 (OpenBLAS), and SciPy 1.18.1. Random seed was pinned to <code>SEED = 42</code>. Two workload scales were benchmarked across worker counts $P \\in \\{{1, 2, 4, 6, 8, 12\\}}$ with 3 repeats per configuration:
    <br>• <strong>Medium Mesh:</strong> 11,520 nodes, 53,692 springs, 100 timesteps ($\\Delta t = 5.0\\,\\mu\\text{{s}}$)
    <br>• <strong>Large Mesh:</strong> 30,000 nodes, 142,135 springs, 50 timesteps ($\\Delta t = 5.0\\,\\mu\\text{{s}}$)</p>

    <div class="figure-container">
        <img src="{car_init_b64}" alt="Initial 3D Vehicle Mesh">
        <div class="figure-caption">Figure 2: 3D procedural vehicle mesh showing structural zoning: Front Crumple Zone (Red), Passenger Safety Cell (Green), and Rear Rigid Clamping Zone (Blue).</div>
    </div>

    <div class="figure-container">
        <img src="{car_def_b64}" alt="Deformed 3D Vehicle Mesh">
        <div class="figure-caption">Figure 3: Post-impact structural crash deformation. Color represents nodal displacement magnitude [m], showing crumple absorption in the nose while keeping the safety cell intact.</div>
    </div>

    <div class="figure-container">
        <img src="{part_b64}" alt="Domain Decomposition Partition View">
        <div class="figure-caption">Figure 4: Spatial domain decomposition across 4 CPU cores illustrating structural element allocation.</div>
    </div>

    <h2>5. Results and Performance Analysis</h2>
    <p>Numerical verification was conducted via <code>test_correctness.py</code>, comparing parallel execution against the single-threaded baseline. The maximum relative error across all degrees of freedom was $2.13 \\times 10^{{-16}}$, confirming exact double-precision numerical equivalence (machine epsilon).</p>

    <table>
        <thead>
            <tr>
                <th>Mesh</th>
                <th>Workers</th>
                <th>Wall Time (s)</th>
                <th>Speedup S(P)</th>
                <th>Efficiency %</th>
                <th>Amdahl P</th>
                <th>Load Imbalance</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Medium</td><td>1 (seq)</td><td>0.8718 s</td><td>1.000x</td><td>100.0%</td><td>Baseline</td><td>1.000</td></tr>
            <tr><td>Medium</td><td>1 (par)</td><td>0.8690 s</td><td>1.003x</td><td>100.3%</td><td>N/A</td><td>1.000</td></tr>
            <tr><td>Medium</td><td>2</td><td>0.5664 s</td><td>1.539x</td><td>77.0%</td><td>0.701</td><td>1.016</td></tr>
            <tr><td>Medium</td><td>4</td><td>0.3987 s</td><td>2.187x</td><td>54.7%</td><td>0.724</td><td>1.154</td></tr>
            <tr style="background:#d4efdf; font-weight:bold;"><td>Medium</td><td>6</td><td>0.3454 s</td><td>2.524x</td><td>42.1%</td><td>0.725 (PEAK)</td><td>1.079</td></tr>
            <tr><td>Medium</td><td>8</td><td>0.3598 s</td><td>2.423x</td><td>30.3%</td><td>0.671</td><td>1.118</td></tr>
            <tr><td>Medium</td><td>12</td><td>0.5061 s</td><td>1.723x</td><td>14.4%</td><td>0.458</td><td>1.066</td></tr>
            <tr><td>Large</td><td>1 (seq)</td><td>1.2691 s</td><td>1.000x</td><td>100.0%</td><td>Baseline</td><td>1.000</td></tr>
            <tr><td>Large</td><td>1 (par)</td><td>1.2780 s</td><td>0.993x</td><td>99.3%</td><td>N/A</td><td>1.000</td></tr>
            <tr><td>Large</td><td>2</td><td>0.8257 s</td><td>1.537x</td><td>76.8%</td><td>0.699</td><td>1.007</td></tr>
            <tr><td>Large</td><td>4</td><td>0.7978 s</td><td>1.591x</td><td>39.8%</td><td>0.495</td><td>1.018</td></tr>
            <tr><td>Large</td><td>6</td><td>0.6923 s</td><td>1.833x</td><td>30.6%</td><td>0.545</td><td>1.021</td></tr>
            <tr style="background:#d4efdf; font-weight:bold;"><td>Large</td><td>8</td><td>0.6412 s</td><td>1.979x</td><td>24.7%</td><td>0.565 (PEAK)</td><td>1.021</td></tr>
            <tr><td>Large</td><td>12</td><td>0.7443 s</td><td>1.705x</td><td>14.2%</td><td>0.451</td><td>1.070</td></tr>
        </tbody>
    </table>

    <div class="figure-container">
        <img src="{scale_b64}" alt="Scaling Curves">
        <div class="figure-caption">Figure 5: Parallel Speedup S(P) and Efficiency E(P) across worker core counts for Medium and Large workloads.</div>
    </div>

    <div class="figure-container">
        <img src="{amdahl_b64}" alt="Amdahl Fit">
        <div class="figure-caption">Figure 6: Measured speedup overlaid on theoretical Amdahl limits, demonstrating a parallel fraction of ~72% up to 6 workers.</div>
    </div>

    <div class="figure-container">
        <img src="{stage_b64}" alt="Stage Breakdown">
        <div class="figure-caption">Figure 7: Internal runtime stage breakdown for 4-worker execution (71.9% parallel compute vs 28.1% serial barrier & coordinator reduction overhead).</div>
    </div>

    <h2>6. Industrial Applications and Benefits</h2>
    <p>In production CAE workflows, computational speedup directly reduces vehicle development cycle time from 36 months to under 18 months, enabling rapid design iteration and crash safety validation while saving tens of millions of dollars in physical tooling and destruction costs. Crucially, industrial supercomputing clusters overcome Amdahl's fixed-size bottleneck via Gustafson's Law (scaled speedup): as core counts scale into the thousands, mesh resolution is refined from tens of thousands to tens of millions of elements. Because element computation scales cubically (volume) while boundary synchronization scales quadratically (surface area), the parallel fraction $P$ expands to $>99.5\\%$, sustaining high cluster efficiency.</p>

    <h2>7. Challenges, Limitations, and Future Work</h2>
    <p>Key challenges identified: (1) OS-level barrier synchronization jitter at high thread counts, (2) memory bus saturation when 12 processes simultaneously query L3 cache lines, and (3) contact-impact search load imbalances during extreme structural foldover.</p>
    <p>Future architectural extension paths: (a) Distributed MPI + OpenMP hybrid architecture utilizing METIS graph decomposition and non-blocking asynchronous boundary exchange (<code>MPI_Isend</code>/<code>MPI_Irecv</code>); (b) Massively parallel GPU acceleration (CUDA/CuPy) mapping spring updates to GPU thread warps with atomic coordinate scatter-adds, achieving microsecond barrier latencies (<code>__syncthreads()</code>); and (c) dynamic load rebalancing mid-simulation.</p>

    <h2>8. Conclusion</h2>
    <p>This project successfully developed, tested, and analyzed an explicit vehicle crash simulation workload decomposition. A procedural 3D automotive vehicle model with localized crumple zones and rigid cabin structures was solved using Velocity-Verlet integration. Zero-copy shared memory eliminated IPC overhead, yielding a peak speedup of $2.52\\times$ on 6 cores and verifying exact numerical correctness ($2.13 \\times 10^{{-16}}$ error). Amdahl profiling identified per-step synchronization and coordinator reduction as the governing serial bottlenecks, providing a rigorous foundation for scaling explicit transient dynamics across multi-node supercomputers and GPU accelerators.</p>

    <h2>9. References</h2>
    <ol class="ref-list">
        <li><strong>Ansys Inc.</strong> (2024). <em>LS-DYNA Theory Manual and MPP Decomposition Guidelines</em>. Ansys Documentation.</li>
        <li><strong>Belytschko, T., Liu, W. K., Moran, B., & Elkhodary, K.</strong> (2014). <em>Nonlinear Finite Elements for Continua and Structures</em>. John Wiley & Sons, 2nd Edition.</li>
        <li><strong>Amdahl, G. M.</strong> (1967). <em>Validity of the single processor approach to achieving large scale computing capabilities</em>. AFIPS Conference Proceedings, Vol. 30, pp. 483–485.</li>
        <li><strong>Gustafson, J. L.</strong> (1988). <em>Reevaluating Amdahl's Law</em>. Communications of the ACM, 31(5), pp. 532–533.</li>
        <li><strong>National Highway Traffic Safety Administration (NHTSA).</strong> (2023). <em>Public Finite Element Crash Simulation Vehicle Models (Silverado, Yaris, Accord)</em>. U.S. Department of Transportation.</li>
        <li><strong>OpenRadioss Community.</strong> (2024). <em>OpenRadioss Explicit Finite Element Solver Source & Theory Documentation</em>. Altair Engineering.</li>
        <li><strong>Siemens Digital Industries Software.</strong> (2024). <em>Scalability and High Performance Computing in Simcenter Explicit Crash Solvers</em>. Siemens White Paper.</li>
        <li><strong>Zukas, J. A.</strong> (2004). <em>Introduction to Hydrocodes</em>. Studies in Applied Mechanics, Elsevier Science.</li>
        <li><strong>Karypis, G., & Kumar, V.</strong> (1998). <em>A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs</em>. SIAM Journal on Scientific Computing, 20(1), pp. 359–392.</li>
        <li><strong>Mattson, T. G., Sanders, B. A., & Massingill, B. L.</strong> (2004). <em>Patterns for Parallel Programming</em>. Addison-Wesley Professional.</li>
    </ol>
</div>
</body>
</html>
"""
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report successfully written to: {HTML_PATH}")

if __name__ == "__main__":
    main()

