"""
test_core_performance.py — Test individual vCPU throughput via taskset
"""
import subprocess, time

print("=" * 60)
print("  vCPU Throughput Test in WSL (taskset -c <cpu>)")
print("=" * 60)

for cpu in range(12):
    # Run a simple compute loop pinned to cpu
    cmd = [
        "wsl", "taskset", "-c", str(cpu),
        "python3", "-c",
        "import time\n"
        "t0 = time.perf_counter()\n"
        "s = 0\n"
        "for i in range(15_000_000): s += i\n"
        "t1 = time.perf_counter()\n"
        "print(f'{(15.0 / (t1 - t0)):.2f}')\n"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    mops = res.stdout.strip()
    print(f"vCPU {cpu:2d} -> {mops} Mops/s")

