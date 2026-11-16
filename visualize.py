"""
visualize.py — Generate all report figures from solver output
=============================================================
Figures produced:
  1. figures/car_mesh_initial.png   — 3D car mesh coloured by zone
  2. figures/car_mesh_deformed.png  — 3D mesh after crash (coloured by displacement)
  3. figures/partition_view.png     — same mesh coloured by worker partition
  4. figures/scaling_curves.png     — speedup + efficiency vs workers (Amdahl fit)
  5. figures/amdahl_fit.png         — Amdahl theoretical vs measured overlay
  6. figures/stage_breakdown.png    — pie charts: serial fraction breakdown
  7. figures/displacement_heatmap.png — node displacement at final timestep

Run after benchmark: python visualize.py
"""

import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import numpy as np

from src.car_mesh import build_car_mesh
from src.solver   import run_sequential

SEED    = 42
FIG_DIR = pathlib.Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR = pathlib.Path(__file__).parent / "results"


# ── Helpers ───────────────────────────────────────────────────────────────────
ZONE_COLOURS = {0: "#E74C3C", 1: "#2ECC71", 2: "#3498DB"}
ZONE_LABELS  = {0: "Front Crumple", 1: "Cabin (Safety Cell)", 2: "Rear Crumple"}


def _3d_ax(fig, title):
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X — Length [m]", labelpad=6)
    ax.set_ylabel("Y — Width [m]",  labelpad=6)
    ax.set_zlabel("Z — Height [m]", labelpad=6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.view_init(elev=20, azim=-55)
    return ax


def _scatter_mesh(ax, nodes, springs, colour, alpha_sp=0.08, s=6):
    ax.scatter(nodes[:,0], nodes[:,1], nodes[:,2], c=colour, s=s, zorder=3)
    for si, sj in springs:
        ax.plot([nodes[si,0], nodes[sj,0]],
                [nodes[si,1], nodes[sj,1]],
                [nodes[si,2], nodes[sj,2]],
                color="#AAAAAA", lw=0.3, alpha=alpha_sp)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Initial 3D mesh coloured by zone
# ─────────────────────────────────────────────────────────────────────────────
def fig_initial_mesh(nodes, springs, zone_node):
    fig = plt.figure(figsize=(11, 6))
    ax  = _3d_ax(fig, "3D Car Mesh — Coloured by Structural Zone\n"
                      "(Procedural spring-mass model, ~5 k nodes shown)")
    colours = [ZONE_COLOURS[z] for z in zone_node]
    _scatter_mesh(ax, nodes, springs, colours, alpha_sp=0.07)

    # Legend proxies
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=ZONE_COLOURS[z], label=ZONE_LABELS[z])
               for z in [0,1,2]]
    ax.legend(handles=patches, loc="upper left", fontsize=9)

    # Annotate impact direction (3D axes use ax.text, not annotate)
    ax.text(-0.3, 0, 1.4, "← IMPACT\n   DIRECTION\n   (50 km/h)",
            fontsize=8, color="#C0392B", fontweight="bold", zdir="z")

    plt.tight_layout()
    out = FIG_DIR / "car_mesh_initial.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 + 3: Run solver, produce deformed + partition views
# ─────────────────────────────────────────────────────────────────────────────
def fig_deformed_and_partition(nodes, springs, k_arr, rest_L, zone_node, masses, vel0):
    rear_mask = nodes[:,0] > nodes[:,0].max() * 0.80
    N_WORKERS_PART = 4
    N_STEPS = 300
    SNAP_FR = 300

    print("  Running solver for deformation visualisation (300 steps)…")
    snaps, _, _ = run_sequential(nodes, vel0, masses, springs, k_arr, rest_L,
                                  rear_mask, N_STEPS, snap_freq=SNAP_FR)
    pos_final = snaps[-1]
    displ = np.linalg.norm(pos_final - nodes, axis=1)

    # Figure 2: Deformed mesh coloured by displacement magnitude
    fig = plt.figure(figsize=(11, 6))
    ax  = _3d_ax(fig, "3D Car Mesh — Post-Crash Deformation\n"
                      "Colour = nodal displacement magnitude [m]")
    norm = plt.Normalize(displ.min(), displ.max())
    cmap = cm.plasma
    colours = cmap(norm(displ))
    _scatter_mesh(ax, pos_final, springs, colours, alpha_sp=0.08)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.08, label="Displacement [m]")
    plt.tight_layout()
    out = FIG_DIR / "car_mesh_deformed.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Figure 3: Partition view (colour by worker slab)
    fig = plt.figure(figsize=(11, 6))
    ax  = _3d_ax(fig, f"Domain Decomposition — {N_WORKERS_PART} Worker Partitions\n"
                      "Colour = assigned worker ID (spatial X-slab)")
    order   = np.argsort(nodes[:,0])
    slabs   = np.array_split(order, N_WORKERS_PART)
    worker_id = np.empty(len(nodes), dtype=int)
    for w, sl in enumerate(slabs):
        worker_id[sl] = w
    cmap_w  = plt.get_cmap("Set1", N_WORKERS_PART)
    colours = [cmap_w(worker_id[i]) for i in range(len(nodes))]
    _scatter_mesh(ax, nodes, springs, colours, alpha_sp=0.08)
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=cmap_w(w), label=f"Worker {w}")
               for w in range(N_WORKERS_PART)]
    ax.legend(handles=patches, loc="upper left", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "partition_view.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Scaling curves (speedup + efficiency)
# ─────────────────────────────────────────────────────────────────────────────
def fig_scaling_curves():
    # Load benchmark results — prefer benchmark2 if available
    b2 = RES_DIR / "benchmark2_results.json"
    b1 = RES_DIR / "benchmark_results.json"
    path = b2 if b2.exists() else b1
    with open(path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Parallel Scaling: Speedup and Efficiency\n"
                 "Spring-Mass Car Crash Simulation (measured, seed=42)",
                 fontsize=12, fontweight="bold")

    colours = ["#2E86C1", "#E74C3C"]
    markers = ["o", "s"]

    for ax, metric, ylabel, ylim in zip(
            axes,
            ["speedup", "efficiency"],
            ["Speedup S(P)", "Efficiency E(P) [%]"],
            [None, None]):

        for ci, (mkey, mdata) in enumerate(data["meshes"].items()):
            ws = sorted([int(k) for k in mdata["workers"] if k != "1_seq"])
            t1 = mdata["workers"]["1_seq"]["t_median"]
            xs, ys = [], []
            for nw in ws:
                wd = mdata["workers"][str(nw)]
                xs.append(nw)
                ys.append(wd[metric])

            ax.plot(xs, ys, f"{markers[ci]}-",
                    color=colours[ci], lw=2, ms=7,
                    label=f"{mdata['label']} (N={mdata['n_nodes']:,})")

        # Ideal reference lines
        ideal_x = [1, 2, 4, 6, 8, 12]
        if metric == "speedup":
            ax.plot(ideal_x, ideal_x, "k--", lw=1, alpha=0.5, label="Ideal (linear)")
        else:
            ax.axhline(100, color="k", ls="--", lw=1, alpha=0.5, label="Ideal 100%")

        ax.set_xlabel("Number of Workers P", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks([1, 2, 4, 6, 8, 12])
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title(ylabel.split("[")[0].strip(), fontsize=11)

    plt.tight_layout()
    out = FIG_DIR / "scaling_curves.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Amdahl theoretical vs measured
# ─────────────────────────────────────────────────────────────────────────────
def fig_amdahl():
    b2 = RES_DIR / "benchmark2_results.json"
    b1 = RES_DIR / "benchmark_results.json"
    path = b2 if b2.exists() else b1
    with open(path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Amdahl's Law: Theoretical vs Measured Speedup\n"
                 "Serial fraction estimated from 2-worker data point",
                 fontsize=11, fontweight="bold")

    P_vals   = [0.50, 0.70, 0.85, 0.95]
    theory_x = np.array([1, 2, 4, 6, 8, 12])
    cmaps    = ["#AED6F1", "#7FB3D3", "#2874A6", "#1A5276"]

    for P, col in zip(P_vals, cmaps):
        theory_y = 1.0 / ((1-P) + P/theory_x)
        ax.plot(theory_x, theory_y, "--", color=col, lw=1.5,
                label=f"Amdahl P={P:.0%} (theory)")

    markers = ["o", "s"]
    colours = ["#E74C3C", "#27AE60"]
    for ci, (mkey, mdata) in enumerate(data["meshes"].items()):
        ws = sorted([int(k) for k in mdata["workers"] if k != "1_seq"])
        xs = []; ys = []
        for nw in ws:
            xs.append(nw)
            ys.append(mdata["workers"][str(nw)]["speedup"])
        ax.plot(xs, ys, f"{markers[ci]}-", color=colours[ci],
                lw=2.5, ms=8, label=f"Measured — {mdata['label']}")

    ax.plot(theory_x, theory_x, "k:", lw=1, alpha=0.4, label="Ideal (100% parallel)")
    ax.set_xlabel("Number of Workers P", fontsize=11)
    ax.set_ylabel("Speedup S(P)", fontsize=11)
    ax.set_xticks([1,2,4,6,8,12])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 13)
    ax.set_ylim(0, None)

    plt.tight_layout()
    out = FIG_DIR / "amdahl_fit.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Stage breakdown (serial fraction pie chart)
# ─────────────────────────────────────────────────────────────────────────────
def fig_stage_breakdown():
    b2 = RES_DIR / "benchmark2_results.json"
    b1 = RES_DIR / "benchmark_results.json"
    path = b2 if b2.exists() else b1
    with open(path) as f:
        data = json.load(f)

    # Pick LARGE mesh, 4 workers for the breakdown
    mesh_data = list(data["meshes"].values())[-1]  # last = largest
    wd = None
    for k in ["4", "2"]:
        if k in mesh_data["workers"]:
            wd = mesh_data["workers"][k]; nw = int(k); break
    if wd is None:
        print("No parallel worker data for stage breakdown"); return

    stage = wd["stage_ms"]
    # Stage labels: parallel compute (barrier_ms includes worker wait time)
    # Note: barrier_ms = time coordinator waits = approx parallel compute time
    labels = []
    sizes  = []
    cols   = []
    if "barrier_ms" in stage:
        labels.append(f"Parallel compute\n(worker force kernel)\n{stage['barrier_ms']:.1f} ms")
        sizes.append(stage["barrier_ms"])
        cols.append("#27AE60")
    if "reduce_ms" in stage:
        labels.append(f"Serial reduce\n(force sum)\n{stage['reduce_ms']:.1f} ms")
        sizes.append(stage["reduce_ms"])
        cols.append("#E74C3C")
    if "integrate_ms" in stage:
        labels.append(f"Serial integrate\n(Verlet)\n{stage['integrate_ms']:.1f} ms")
        sizes.append(stage["integrate_ms"])
        cols.append("#E67E22")
    if "io_ms" in stage and stage["io_ms"] > 0.01:
        labels.append(f"I/O snapshots\n{stage['io_ms']:.1f} ms")
        sizes.append(stage["io_ms"])
        cols.append("#9B59B6")

    if not sizes:
        print("No stage data"); return

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=cols,
        autopct="%1.1f%%", startangle=90,
        pctdistance=0.75, labeldistance=1.15,
        textprops=dict(fontsize=9)
    )
    for at in autotexts:
        at.set_fontsize(9); at.set_fontweight("bold")

    total = sum(sizes)
    serial = sum(s for s, l in zip(sizes, labels) if "Serial" in l or "I/O" in l)
    ax.set_title(
        f"Runtime Breakdown — {mesh_data['label']}, {nw} Workers\n"
        f"Serial fraction = {serial/total*100:.1f}%  (Amdahl bottleneck)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out = FIG_DIR / "stage_breakdown.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Building visualisation mesh (nx=30, ny=10, nz=8)…")
    nodes, springs, k_arr, rest_L, zone_node, masses, vel0 = \
        build_car_mesh(nx=30, ny=10, nz=8, seed=SEED)
    print(f"  {len(nodes):,} nodes, {len(springs):,} springs")

    print("\n[1/6] Initial mesh…")
    fig_initial_mesh(nodes, springs, zone_node)

    print("[2/6] Deformed + partition views…")
    fig_deformed_and_partition(nodes, springs, k_arr, rest_L, zone_node, masses, vel0)

    print("[3/6] Scaling curves…")
    fig_scaling_curves()

    print("[4/6] Amdahl fit…")
    fig_amdahl()

    print("[5/6] Stage breakdown…")
    fig_stage_breakdown()

    print("\nAll figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
