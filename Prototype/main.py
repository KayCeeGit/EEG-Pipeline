import nbformat
from nbconvert import PythonExporter
import runpy
import tempfile
import os
import tracemalloc
import psutil
import json
import time
from datetime import datetime

def run_notebook(nb_path,start_time):
    
    exporter = PythonExporter()
    code, _ = exporter.from_filename(nb_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
        tmp.write(code.encode("utf-8"))
        tmp_path = tmp.name

    performance = psutil.Process(os.getpid())
    cpuAtStart = performance.cpu_times()
    memAtStart = performance.memory_info().rss / (1024 * 1024)
    tracemalloc.start()

    # Run notebook code
    runpy.run_path(tmp_path, run_name="__main__")

    cpuAtEnd = performance.cpu_times()
    memAtEnd = performance.memory_info().rss / (1024 * 1024)

    #memory metrics
    current, peak = tracemalloc.get_traced_memory()
    memPeakInMB = peak / (1024 * 1024)  
    tracemalloc.stop()

    #time metrics
    elapsed = time.time() - start_time

    #cpu metrics
    cpuUserTime = cpuAtEnd.user - cpuAtStart.user
    cpuSystemTime = cpuAtEnd.system - cpuAtStart.system
    cpuTotalTime = cpuUserTime + cpuSystemTime
    
    
    cpu_utilization_ratio = cpuTotalTime / (elapsed * psutil.cpu_count(logical=True))

    summary = {
        "architecture": "Prototype",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_time_sec": elapsed,
        "cpu_user_time": cpuUserTime,
        "cpu_system_time": cpuSystemTime,
        "cpu_utilization_ratio": round(cpu_utilization_ratio, 3),  
        "mem_before_MB": memAtStart,
        "mem_after_MB": memAtEnd,
        "mem_peak_MB": memPeakInMB,
        "mem_diff_MB": memAtEnd - memAtStart,
    }


    summary_path = "results/Prototype_run_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []
    existing.append(summary)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4)

if __name__ == "__main__":
    start_time = time.time()

    nb_path = os.path.join(os.path.dirname(__file__), "prototype.ipynb")
    run_notebook(nb_path,start_time)
