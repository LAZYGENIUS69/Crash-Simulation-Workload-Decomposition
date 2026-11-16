"""
dep_graph.py — Task Dependency Graph for the crash simulation pipeline
=======================================================================
Generates figures/dep_graph.png showing:
  - Task nodes (computational kernels)
  - Dependency edges (data flow / ordering constraints)
  - Colour coding: parallel tasks (blue) vs serial tasks (red)
  - Per-timestep loop structure

This graph directly maps to the rubric requirement:
  "task dependency graph; identify independent tasks, dependencies,
   and synchronization points"

Industrial analogy:
  Node labels map to LS-DYNA MPP task graph (internal force assembly,
  contact search, global time-step, output).

Run: python dep_graph.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

FIG_DIR = pathlib.Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Task node definitions ─────────────────────────────────────────────────────
# (id, label, type, description)
# type: "init", "parallel", "serial", "io", "sync"
TASKS = [
    # Initialisation phase (once)
    ("MESH_GEN",    "Mesh\nGeneration",          "init"),
    ("PARTITION",   "Domain\nPartition",          "init"),
    ("SPRING_ASGN", "Spring\nAssignment\n(to workers)", "init"),

    # Per-timestep loop — parallel phase
    ("W0_FORCE",  "Worker 0\nForce\nCompute",   "parallel"),
    ("W1_FORCE",  "Worker 1\nForce\nCompute",   "parallel"),
    ("Wn_FORCE",  "Worker P-1\nForce\nCompute", "parallel"),

    # Synchronisation point
    ("BARRIER",   "Barrier\n(MPI_Barrier\nanalogue)", "sync"),

    # Serial phase — coordinator
    ("REDUCE",    "Force\nReduce\n(sum)",        "serial"),
    ("VERLET",    "Verlet\nIntegration",         "serial"),
    ("CLAMP",     "Boundary\nConditions\n(rear clamp)", "serial"),

    # Output / next step
    ("SNAPSHOT",  "Snapshot\nI/O\n(every 10 steps)", "io"),
    ("NEXT_STEP", "Next\nTimestep",              "sync"),

    # Exit condition
    ("END",       "END\n(t > t_final)",          "init"),
]

EDGES = [
    # Init chain
    ("MESH_GEN",   "PARTITION"),
    ("PARTITION",  "SPRING_ASGN"),

    # Spawn workers (all from SPRING_ASGN)
    ("SPRING_ASGN", "W0_FORCE"),
    ("SPRING_ASGN", "W1_FORCE"),
    ("SPRING_ASGN", "Wn_FORCE"),

    # Workers → barrier
    ("W0_FORCE",  "BARRIER"),
    ("W1_FORCE",  "BARRIER"),
    ("Wn_FORCE",  "BARRIER"),

    # Barrier → coordinator serial pipeline
    ("BARRIER",   "REDUCE"),
    ("REDUCE",    "VERLET"),
    ("VERLET",    "CLAMP"),

    # Optional I/O
    ("CLAMP",     "SNAPSHOT"),

    # Loop back
    ("CLAMP",     "NEXT_STEP"),
    ("SNAPSHOT",  "NEXT_STEP"),
    ("NEXT_STEP", "W0_FORCE"),
    ("NEXT_STEP", "W1_FORCE"),
    ("NEXT_STEP", "Wn_FORCE"),
    ("NEXT_STEP", "END"),
]

COLOUR = {
    "init":     "#4A90D9",   # blue
    "parallel": "#27AE60",   # green
    "serial":   "#E74C3C",   # red
    "sync":     "#F39C12",   # amber
    "io":       "#9B59B6",   # purple
}

SHAPE_STYLE = {
    "init":     dict(boxstyle="round,pad=0.4", facecolor="#D6EAF8", edgecolor="#2E86C1", lw=2),
    "parallel": dict(boxstyle="round,pad=0.4", facecolor="#D5F5E3", edgecolor="#1E8449", lw=2),
    "serial":   dict(boxstyle="round,pad=0.4", facecolor="#FADBD8", edgecolor="#C0392B", lw=2),
    "sync":     dict(boxstyle="round,pad=0.4", facecolor="#FDEBD0", edgecolor="#E67E22", lw=2.5),
    "io":       dict(boxstyle="round,pad=0.4", facecolor="#E8DAEF", edgecolor="#7D3C98", lw=2),
}


def build_graph():
    G = nx.DiGraph()
    type_map = {t[0]: t[2] for t in TASKS}
    label_map = {t[0]: t[1] for t in TASKS}

    for tid, label, ttype in TASKS:
        G.add_node(tid, label=label, ttype=ttype)
    for src, dst in EDGES:
        G.add_edge(src, dst)
    return G, type_map, label_map


def manual_layout():
    """Hand-crafted layout for readability."""
    pos = {
        # Init column (left)
        "MESH_GEN":    (0.0, 4.5),
        "PARTITION":   (0.0, 3.5),
        "SPRING_ASGN": (0.0, 2.5),

        # Parallel workers (centre-top)
        "W0_FORCE":    (-1.8, 1.0),
        "W1_FORCE":    ( 0.0, 1.0),
        "Wn_FORCE":    ( 1.8, 1.0),

        # Sync
        "BARRIER":     ( 0.0, 0.0),

        # Serial coordinator
        "REDUCE":      ( 0.0,-1.0),
        "VERLET":      ( 0.0,-2.0),
        "CLAMP":       ( 0.0,-3.0),

        # I/O & control
        "SNAPSHOT":    ( 2.2,-3.0),
        "NEXT_STEP":   ( 0.0,-4.2),
        "END":         ( 2.2,-4.2),
    }
    return pos


def draw(G, type_map, label_map, pos):
    fig, ax = plt.subplots(figsize=(12, 14), facecolor="#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    ax.set_xlim(-3.2, 3.8)
    ax.set_ylim(-5.2, 5.5)
    ax.axis("off")

    # Draw edges first (behind nodes)
    # Separate: loop-back edges (curved) vs forward edges (straight)
    loop_edges  = [("NEXT_STEP","W0_FORCE"), ("NEXT_STEP","W1_FORCE"),
                   ("NEXT_STEP","Wn_FORCE")]
    normal_edges = [(u,v) for u,v in G.edges() if (u,v) not in loop_edges]

    nx.draw_networkx_edges(
        G, pos, edgelist=normal_edges, ax=ax,
        edge_color="#555555", arrows=True,
        arrowstyle="-|>", arrowsize=18,
        width=1.8, min_source_margin=22, min_target_margin=22,
        connectionstyle="arc3,rad=0.0"
    )
    # Loop-back edges (curved)
    nx.draw_networkx_edges(
        G, pos, edgelist=loop_edges, ax=ax,
        edge_color="#888888", arrows=True,
        arrowstyle="-|>", arrowsize=14,
        width=1.4, style="dashed",
        min_source_margin=22, min_target_margin=22,
        connectionstyle="arc3,rad=0.3"
    )

    # Draw nodes as text boxes
    for node_id, (x, y) in pos.items():
        ttype = type_map[node_id]
        label = label_map[node_id]
        style = SHAPE_STYLE[ttype]
        ax.text(
            x, y, label,
            ha="center", va="center", fontsize=9, fontweight="bold",
            multialignment="center",
            bbox=dict(**style, linewidth=style["lw"]),
            zorder=5
        )

    # ── Annotation zone bands ────────────────────────────────────────
    # "PARALLEL ZONE" shaded background
    ax.axhspan(0.3, 1.8, xmin=0.05, xmax=0.95,
               facecolor="#D5F5E3", alpha=0.35, zorder=0)
    ax.text(-2.9, 1.05, "PARALLEL\nZONE\n(P workers)", fontsize=8,
            color="#1E8449", va="center", style="italic")

    # "SERIAL ZONE" shaded background
    ax.axhspan(-3.6, -0.3, xmin=0.05, xmax=0.95,
               facecolor="#FADBD8", alpha=0.30, zorder=0)
    ax.text(-2.9, -1.8, "SERIAL\nZONE\n(coordinator)", fontsize=8,
            color="#C0392B", va="center", style="italic")

    # Sync point label
    ax.annotate("SYNCHRONISATION\nPOINT", xy=(0.0, 0.0), xytext=(2.6, 0.0),
                fontsize=8, color="#E67E22", va="center",
                arrowprops=dict(arrowstyle="->", color="#E67E22", lw=1.5),
                bbox=dict(boxstyle="round", facecolor="#FDEBD0", edgecolor="#E67E22"))

    # ── Legend ───────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(facecolor=COLOUR["init"],     label="Initialisation (once)"),
        mpatches.Patch(facecolor=COLOUR["parallel"], label="Parallel task (per-step)"),
        mpatches.Patch(facecolor=COLOUR["serial"],   label="Serial task (coordinator)"),
        mpatches.Patch(facecolor=COLOUR["sync"],     label="Synchronisation / Barrier"),
        mpatches.Patch(facecolor=COLOUR["io"],       label="I/O / Snapshot"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9,
              framealpha=0.9, edgecolor="#BBBBBB")

    ax.set_title(
        "Task Dependency Graph — Crash-Simulation Workload Decomposition\n"
        "(Shared-Memory Spring-Force Domain Decomposition, P workers)",
        fontsize=13, fontweight="bold", pad=10
    )

    # Timestep loop label
    ax.text(-2.9, -4.2, "← TIMESTEP LOOP\n   (repeats n_steps times)",
            fontsize=8, color="#555555", va="center", style="italic")

    plt.tight_layout()
    out = FIG_DIR / "dep_graph.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return out


def main():
    G, type_map, label_map = build_graph()
    pos = manual_layout()
    out = draw(G, type_map, label_map, pos)

    # Also print adjacency for the report
    print("\nTask adjacency (dependency list):")
    print(f"  {'Source':<15} → {'Target'}")
    for u, v in G.edges():
        print(f"  {u:<15} → {v}")

    # Identify parallel vs serial tasks
    par = [n for n, d in G.nodes(data=True) if d.get("ttype") == "parallel"]
    ser = [n for n, d in G.nodes(data=True) if d.get("ttype") in ("serial","sync")]
    print(f"\nIndependent parallel tasks (per step): {par}")
    print(f"Serial / sync tasks: {ser}")


if __name__ == "__main__":
    main()

