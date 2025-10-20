import argparse
import json
import numpy as np
from tqdm import tqdm
import time
import psutil
import os
import tracemalloc

from Microkernel.microkernel import Kernel

from Microkernel.plugins.dataloader import DataLoadPlugin
from Microkernel.plugins.labelloader import LabelLoadPlugin
from Microkernel.plugins.preprocess import PreprocessPlugin
from Microkernel.plugins.segmentation import SegmentationPlugin
from Microkernel.plugins.featureExtraction import FeatureExtractionPlugin
from Microkernel.plugins.modelPLugin import ModelPlugin

if __name__ == "__main__": 
    tracemalloc.start()     
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    RANDOMSEED = args.seed
    
    start_time = time.time()
    performance = psutil.Process(os.getpid())
    cpuAtStart = performance.cpu_times()
    memAtStart = performance.memory_info().rss / (1024 * 1024)

    allX = []
    allY = []
    
    preprocessConfig = {
        "notch": True,
        "bandpass": True,
        "downsample": True,
        "ica": True,
        "downsample_rate": 128,
        "lFreq": 1.5,
        "hFreq": 45.0,
        "icaMethod": "fastica",
        "randSeed": RANDOMSEED,
        "n_components": 0.999999,
    }
    
    
    segmenationConfig = {
        "windowSize": 4.0,
        "overlap": 0.75,
    }
    
    modelConfigs = [
    {"modelType": "SVM", "C": 0.1, "kernel": "linear", "lda": True},
    {"modelType": "SVM", "C": 10, "kernel": "linear", "lda": True},
    
    {"modelType": "SVM", "C": 0.1, "kernel": "linear", "lda": False},
    {"modelType": "SVM", "C": 10, "kernel": "linear", "lda": False},
    
    {"modelType": "KNN", "n_neighbors": 5},
    {"modelType": "KNN", "n_neighbors": 7},
    {"modelType": "KNN", "n_neighbors": 25},
    
    {"modelType": "RF", "n_estimators": 100},
    {"modelType": "RF", "n_estimators": 500},
    
    {"modelType": "MLP", "activation": "relu"},
    {"modelType": "MLP", "activation": "tanh"},
    {"modelType": "MLP", "activation": "logistic"},
]
    
    kernel = Kernel()

    #add plugins to kernal
    kernel.registerPlugin(DataLoadPlugin(), "dataloader")
    kernel.registerPlugin(LabelLoadPlugin(), "labelloader")
    kernel.registerPlugin(PreprocessPlugin(config= preprocessConfig), "preprocess")
    kernel.registerPlugin(SegmentationPlugin(config= segmenationConfig), "segmentation")
    kernel.registerPlugin(FeatureExtractionPlugin(), "featureExtraction")


    for i in tqdm(range(1, 3)): 
        data = {
            "filepath": f"s{i:02d}.bdf",
            "excelFile": "participant_ratings.xls",
            "participant": i,
            "eventId": 4,  
        }
        result = kernel.execute(data) # runs all the plugins in the order above. All the data is put into the allx and ally arrays for models, similar to other architectures
        allX.append(result["features"])
        allY.append(result["ySegments"])
        
       
    allX = np.vstack(allX)
    allY = np.hstack(allY)
    XandY = {"features": allX, "labels": allY}
    
    kernel = Kernel() #reset kernal to clear previous plugins 
    modelPlugin = ModelPlugin(random_seed=RANDOMSEED)
    kernel.registerPlugin(modelPlugin)
    
    logs = []
    for config in modelConfigs:
        print(f"\n===== Running {config} Model =====")
        modelPlugin.config = config  
        metrics = kernel.execute(XandY)
        print(f"{metrics}")
        log = {
            "architecture": "Microkernel",
            "model": config,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "report": metrics.get("report"),
        }
        logs.append(log)


    os.makedirs("results", exist_ok=True)
    logPath = os.path.join("results", "Microkernel_model_results.json")

    if os.path.exists(logPath):
        with open(logPath, "r") as f:
            old_data = json.load(f)
    else:
        old_data = []

    old_data.extend(logs)

    with open(logPath, "w") as f:
        json.dump(old_data, f, indent=4)
        
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
    cpuRatio = 0.0 #used to calculate cpu utilization ratio
    
    if elapsed > 0:
        cpu_utilization_ratio = cpuTotalTime / (elapsed * psutil.cpu_count(logical=True))

    summary = {
        "architecture": "Layered",
        "elapsed_time_sec": elapsed,
        "cpu_user_time": cpuUserTime,
        "cpu_system_time": cpuSystemTime,
        "cpu_utilization_ratio": round(cpu_utilization_ratio, 3),  
        "mem_before_MB": memAtStart,
        "mem_after_MB": memAtEnd,
        "mem_peak_MB": memPeakInMB,
        "mem_diff_MB": memAtEnd - memAtStart,
    }

    # write or append to JSON file
    os.makedirs("results", exist_ok=True)
    perf_path = os.path.join("results", "Microkernal_mem_sums.json")
    if os.path.exists(perf_path):
        with open(perf_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.append(summary)
    with open(perf_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4)
