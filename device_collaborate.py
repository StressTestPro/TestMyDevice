import time
import threading
import psutil
import subprocess
import platform
import datetime

# Optional imports for GPU test
try:
    from numba import cuda
    import numpy as np
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

import speedtest


def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def cpu_stress_test(duration_sec=10, num_threads=4):
    def cpu_load():
        while not stop_event.is_set():
            pass

    stop_event = threading.Event()
    threads = []

    log(f"Starting CPU stress test for {duration_sec}s on {num_threads} threads...")
    for _ in range(num_threads):
        t = threading.Thread(target=cpu_load)
        t.start()
        threads.append(t)

    start_time = time.time()
    while time.time() - start_time < duration_sec:
        usage = psutil.cpu_percent(interval=1)
        log(f"CPU Usage: {usage}%")

    stop_event.set()
    for t in threads:
        t.join()

    log("CPU stress test completed.")


def ram_stress_test(size_mb=500, duration_sec=10):
    log(f"Starting RAM stress test: allocating ~{size_mb}MB for {duration_sec}s...")
    try:
        block = bytearray(size_mb * 1024 * 1024)
        for i in range(0, len(block), 4096):
            block[i] = 1

        log("RAM allocated and touched.")
        for _ in range(duration_sec):
            usage = psutil.virtual_memory().percent
            log(f"RAM Usage: {usage}%")
            time.sleep(1)

    except MemoryError:
        log("Memory allocation failed! Try smaller size.")
    finally:
        del block
        log("RAM stress test completed, memory released.")


def gpu_stress_test(duration_sec=10):
    if not GPU_AVAILABLE:
        log("GPU stress test skipped: numba or CUDA not available.")
        return

    n = 10 ** 7
    arr = np.ones(n, dtype=np.float32)
    d_arr = cuda.to_device(arr)

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    @cuda.jit
    def gpu_stress_kernel(arr):
        idx = cuda.grid(1)
        if idx < arr.size:
            val = arr[idx]
            for _ in range(1000):
                val = (val * 1.0001 + 0.0001) % 1000
            arr[idx] = val

    log(f"Starting GPU stress test for {duration_sec} seconds...")
    start = time.time()
    while time.time() - start < duration_sec:
        gpu_stress_kernel[blocks, threads_per_block](d_arr)
        cuda.synchronize()
        log("GPU kernel iteration completed")

    log("GPU stress test completed.")


def ping_test(host="8.8.8.8", count=4):
    log(f"Pinging {host} ({count} times)...")
    param = "-n" if platform.system().lower() == "windows" else "-c"
    command = ["ping", param, str(count), host]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    print(result.stdout)


def internet_speed_test():
    log("Running internet speed test...")
    st = speedtest.Speedtest()
    download_speed = st.download() / 1_000_000  # Mbps
    upload_speed = st.upload() / 1_000_000      # Mbps
    log(f"Download Speed: {download_speed:.2f} Mbps")
    log(f"Upload Speed: {upload_speed:.2f} Mbps")


def main():
    log("=== System Stress and Network Test Start ===")

    cpu_stress_test(duration_sec=15, num_threads=psutil.cpu_count(logical=True))
    ram_stress_test(size_mb=1000, duration_sec=15)
    gpu_stress_test(duration_sec=15)  # Comment this out if no GPU/CUDA

    ping_test()
    internet_speed_test()

    log("=== All tests complete! ===")
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
