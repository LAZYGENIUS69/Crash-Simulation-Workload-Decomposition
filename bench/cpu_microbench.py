import time
import os

print("=== MEASURING PER-vCPU EXECUTION THROUGHPUT IN WSL2 ===")
rates = []
for cpu in range(12):
    os.sched_setaffinity(0, {cpu})
    # Warmup
    x = 1.0
    for _ in range(500_000):
        x = (x * 1.000001 + 0.000002) * 0.999999
    # Measure
    t0 = time.perf_counter()
    for _ in range(10_000_000):
        x = (x * 1.000001 + 0.000002) * 0.999999
    elapsed = time.perf_counter() - t0
    rate = 10.0 / elapsed
    rates.append(rate)
    print(f"  vCPU {cpu:2d} | 10M float ops: {elapsed:.4f} s | Rate: {rate:5.2f} Mops/s")

avg_p = sum(rates[:4]) / 4.0
avg_e = sum(rates[4:]) / 8.0
print(f"\nAverage Throughput: vCPUs 0-3 (Nominal P-cores) = {avg_p:.2f} Mops/s")
print(f"Average Throughput: vCPUs 4-11 (Nominal E-cores) = {avg_e:.2f} Mops/s")
print(f"Throughput Ratio (P / E) = {avg_p / avg_e:.2f}x")

