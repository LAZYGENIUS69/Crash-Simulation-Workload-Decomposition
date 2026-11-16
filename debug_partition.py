"""
debug_partition.py — inspect ghost node counts to fix ghost buffer design
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.car_mesh import build_car_mesh
from src.solver import _partition
import numpy as np

nodes, springs, k_arr, rest_L, zone_node, masses, vel0 = \
    build_car_mesh(nx=10, ny=4, nz=4)
N = len(nodes)
print(f"N={N} nodes, M={len(springs)} springs")

for n_workers in [2, 4]:
    workers, slab_of = _partition(nodes[:,0], springs, n_workers)
    print(f"\nn_workers={n_workers}")
    for w, wd in enumerate(workers):
        print(f"  worker {w}: slab={len(wd['slab'])} nodes, "
              f"left_ghost={len(wd['left_global'])}, right_ghost={len(wd['right_global'])}, "
              f"springs={wd['my_sp_mask'].sum()}")

