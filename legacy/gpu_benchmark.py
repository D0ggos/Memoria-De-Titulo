"""
GPU stress test / benchmark para RTX 3050 (6 GB).
- Compute: matmul FP32, TF32 y FP16 -> TFLOPS
- Memoria: ancho de banda VRAM (copia)
- Térmico: carga sostenida muestreando temp/potencia/reloj con nvidia-smi

Diseñado para saturar la GPU al 100% sin quedarse sin VRAM (6 GB).
"""
import time
import subprocess
import threading
import statistics
import torch


def sample_nvidia_smi(stop_evt, samples):
    """Muestrea temp, potencia, reloj y util cada 0.5 s en un hilo aparte."""
    q = ("--query-gpu=temperature.gpu,power.draw,clocks.sm,"
         "utilization.gpu,memory.used")
    while not stop_evt.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", q, "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL).strip()
            parts = [p.strip() for p in out.split(",")]
            samples.append(tuple(float(x) for x in parts))
        except Exception:
            pass
        stop_evt.wait(0.5)


def matmul_bench(dtype, n, iters, tf32=False):
    """Mide TFLOPS de un matmul n x n sostenido."""
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    # warmup
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    flops = 2.0 * n**3 * iters
    tflops = flops / dt / 1e12
    del a, b, c
    torch.cuda.empty_cache()
    return tflops, dt


def mem_bandwidth(mb=1024, iters=200):
    """Ancho de banda de VRAM vía copias grandes."""
    n = mb * 1024 * 1024 // 4  # float32
    a = torch.randn(n, device="cuda")
    b = torch.empty_like(a)
    for _ in range(10):
        b.copy_(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    bytes_moved = 2 * a.numel() * 4 * iters  # read + write
    gbps = bytes_moved / dt / 1e9
    del a, b
    torch.cuda.empty_cache()
    return gbps


def sustained_stress(seconds, n=8192):
    """Carga sostenida FP16 para test térmico durante `seconds`."""
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    t0 = time.perf_counter()
    count = 0
    while time.perf_counter() - t0 < seconds:
        for _ in range(50):
            c = a @ b
        torch.cuda.synchronize()
        count += 50
    dt = time.perf_counter() - t0
    tflops = 2.0 * n**3 * count / dt / 1e12
    del a, b, c
    torch.cuda.empty_cache()
    return tflops, count


def line(w=64):
    print("=" * w)


def main():
    line()
    print("  GPU BENCHMARK / STRESS TEST")
    line()
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA disp.   : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("!! CUDA no disponible. Abortando.")
        return
    dev = torch.cuda.get_device_properties(0)
    print(f"GPU          : {dev.name}")
    print(f"VRAM total   : {dev.total_memory/1e9:.2f} GB")
    print(f"SMs          : {dev.multi_processor_count}")
    print(f"Compute cap  : {dev.major}.{dev.minor}")
    print(f"CUDA runtime : {torch.version.cuda}")
    line()

    # --- COMPUTE ---
    print("\n[1/4] COMPUTE - matmul (TFLOPS, mas alto = mejor)")
    tf32_flops, _ = matmul_bench(torch.float32, 4096, 200, tf32=False)
    print(f"  FP32 (4096)      : {tf32_flops:7.2f} TFLOPS")
    tf32_on, _ = matmul_bench(torch.float32, 4096, 200, tf32=True)
    print(f"  TF32 (4096)      : {tf32_on:7.2f} TFLOPS")
    fp16_flops, _ = matmul_bench(torch.float16, 4096, 300, tf32=False)
    print(f"  FP16 (4096)      : {fp16_flops:7.2f} TFLOPS")

    # --- MEMORIA ---
    print("\n[2/4] MEMORIA - ancho de banda VRAM")
    bw = mem_bandwidth(1024, 300)
    print(f"  Bandwidth        : {bw:7.1f} GB/s")

    # --- TERMICO SOSTENIDO ---
    dur = 180  # 3 min de carga continua
    print(f"\n[3/4] TERMICO - carga sostenida {dur}s (esto calienta la GPU)")
    stop = threading.Event()
    samples = []
    th = threading.Thread(target=sample_nvidia_smi, args=(stop, samples))
    th.start()
    st_flops, iters = sustained_stress(dur)
    stop.set()
    th.join()
    print(f"  FP16 sostenido   : {st_flops:7.2f} TFLOPS ({iters} matmuls)")

    # --- REPORTE TERMICO ---
    print("\n[4/4] REPORTE TERMICO / ENERGIA (durante carga sostenida)")
    if samples:
        temps = [s[0] for s in samples]
        pwr = [s[1] for s in samples]
        clk = [s[2] for s in samples]
        util = [s[3] for s in samples]
        mem = [s[4] for s in samples]
        print(f"  Temp    max/prom : {max(temps):.0f} / "
              f"{statistics.mean(temps):.0f} C")
        print(f"  Potencia max/prom: {max(pwr):.1f} / "
              f"{statistics.mean(pwr):.1f} W")
        print(f"  Reloj SM prom    : {statistics.mean(clk):.0f} MHz")
        print(f"  Util GPU prom    : {statistics.mean(util):.0f} %")
        print(f"  VRAM usada max   : {max(mem):.0f} MB")
        print(f"  Muestras         : {len(samples)}")
    line()
    print("  BENCHMARK COMPLETO")
    line()


if __name__ == "__main__":
    main()
