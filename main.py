import platform
import multiprocessing
import subprocess
import time

import numpy as np
import pandas as pd


def system_info():
    print("=" * 56)
    print("SYSTEM")
    print("=" * 56)
    print(f"  Python:    {platform.python_version()}")
    print(f"  Machine:   {platform.machine()}")

    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        ).stdout.strip() or "Apple Silicon"
        print(f"  Chip:      {chip}")
    except Exception:
        pass

    try:
        ram_bytes = int(subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True
        ).stdout.strip())
        print(f"  RAM:       {ram_bytes / 1024**3:.0f} GB")
    except Exception:
        pass

    print(f"  CPU cores: {multiprocessing.cpu_count()} logical")
    print()


def run(label, fn):
    print(f"  {label} ... ", end="", flush=True)
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.3f}s")
    return elapsed, result


def bench_cpu_single():
    # Sieve of Eratosthenes — pure Python, single core
    n = 10_000_000
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return sum(sieve)


def bench_matmul():
    # Matrix multiply — exercises Apple Accelerate BLAS (AMX coprocessor on M-series)
    size = 4096
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    C = A @ B
    return float(C.sum()), size


def bench_memory_bandwidth():
    # Large sequential array copy — measures unified memory bandwidth
    n = 250_000_000  # 1 GB of float32
    a = np.ones(n, dtype=np.float32)
    b = np.empty_like(a)
    np.copyto(b, a)
    return float(b.sum()), n


def bench_pandas():
    # Large groupby aggregation — CPU + memory together
    n = 10_000_000
    df = pd.DataFrame({
        "sensor_id": np.random.randint(0, 200, n),
        "temperature": np.random.uniform(20.0, 80.0, n),
        "humidity": np.random.uniform(30.0, 90.0, n),
        "pressure": np.random.uniform(950.0, 1050.0, n),
    })
    result = df.groupby("sensor_id").agg(["mean", "min", "max", "std"])
    return result.shape


def main():
    system_info()

    print("=" * 56)
    print("BENCHMARKS")
    print("=" * 56)

    t_cpu, _ = run("CPU single-core  (prime sieve, n=10M)", bench_cpu_single)
    t_mm, (_, size) = run("NumPy matmul     (float32 4096×4096)  ", bench_matmul)
    t_bw, (_, n) = run("Memory bandwidth (float32, 1 GB copy)  ", bench_memory_bandwidth)
    t_pd, _ = run("Pandas groupby   (10M rows, 200 groups)", bench_pandas)

    gflops = 2 * size ** 3 / t_mm / 1e9
    gb_s = (n * 4 * 2) / t_bw / 1e9  # read + write

    print()
    print("=" * 56)
    print("SUMMARY")
    print("=" * 56)
    print(f"  CPU single-core:   {t_cpu:.3f}s")
    print(f"  Matrix multiply:   {t_mm:.3f}s   →  {gflops:.1f} GFLOPS")
    print(f"  Memory bandwidth:  {t_bw:.3f}s   →  {gb_s:.1f} GB/s")
    print(f"  Pandas groupby:    {t_pd:.3f}s")
    print()


if __name__ == "__main__":
    main()
