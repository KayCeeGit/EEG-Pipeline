import argparse
import json
import numpy as np
from tqdm import tqdm
import time
import psutil
import os
import tracemalloc

from Pipes_And_Filters.Filters.dataload.eegDataLoad import EegDataLoad
from Pipes_And_Filters.Filters.dataload.labelLoad import LabelLoad

from Pipes_And_Filters.Filters.preprocess.notchFilter import NotchFilter
from Pipes_And_Filters.Filters.preprocess.bandpass import Bandpass
from Pipes_And_Filters.Filters.preprocess.downsample import Downsample
from Pipes_And_Filters.Filters.preprocess.ica import Ica

from Pipes_And_Filters.Filters.segment.segment import Segment

from Pipes_And_Filters.Filters.featureExtraction.featureExtract import FeatureExtract

from Pipes_And_Filters.Filters.controller import Controller
from Pipes_And_Filters.data import Data

from Pipes_And_Filters.Filters.models.models import SVMModel, KNNModel, RandomForestModel, MLPModel
from Pipes_And_Filters.Filters.models.modelFilter import ModelFilter

tracemalloc.start()
start_time = time.time()
performance = psutil.Process(os.getpid())
cpuAtStart = performance.cpu_times()
memAtStart = performance.memory_info().rss / (1024 * 1024)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
np.random.seed(args.seed)
RANDOMSEED = args.seed

filters = [
    (EegDataLoad(), {}),
    (LabelLoad(), {"excelFile": "participant_ratings.xls", "participant": 1}),
    (NotchFilter(), {}),
    (Bandpass(), {"lfreq": 1.5, "hfreq": 45.0}),
    (Downsample(downsamplerate=128), {}),
    (Ica(), {"icaMethod": "fastica", "randSeed": RANDOMSEED, "nComponents": 0.999999}),
    (Segment(eventID= 4),{}),  
    (FeatureExtract(), {"samplingRate": 128}),
]

models = [
    ("SVMlda-0.1", SVMModel(C=0.1, kernel="linear", lda=True)),
    ("SVMlda-10", SVMModel(C=10, kernel="linear", lda=True)),
    ("SVMnoLda-0.1", SVMModel(C=0.1, kernel="linear", lda=False)),
    ("SVMnoLda-10", SVMModel(C=10, kernel="linear", lda=False)),
    ("KNN-5", KNNModel(n_neighbors=5)),
    ("KNN-7", KNNModel(n_neighbors=7)),
    ("KNN-25", KNNModel(n_neighbors=25)),
    ("RF-100", RandomForestModel(nEstimators=100)),
    ("RF-500", RandomForestModel(nEstimators=500)),
    ("MLPrelu", MLPModel(activation="relu")),
    ("MLPtanh", MLPModel(activation="tanh")),
    ("MLPlogistic", MLPModel(activation="logistic")),
]


pipe = Controller(filters)
allX = []
allY = []

for i in tqdm(range(1, 3)):
    filepath = f"s{i:02d}.bdf"
    packet = Data(filepath=filepath, excelFile="participant_ratings.xls", participant=i)
    output = pipe.run(packet)
    allX.append(output.features)
    allY.append(output.ySegments)
    
allX = np.vstack(allX)
allY = np.hstack(allY)

for name, model in models:
    print(f"\n===== Running {name} =====")
    results = model.fit(allX, allY)
    log = {
            "architecture": "PipesAndFilters",
            "model": name,
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
            "report": results["report"],
        }

    os.makedirs("results", exist_ok=True)
    logPath = os.path.join("results", f"Pipes_And_Filters_model_results.json")

    # Append instead of overwriting
    if os.path.exists(logPath):
        with open(logPath, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(log)
    with open(logPath, "w") as f:
        json.dump(data, f, indent=4)
    # print(f"Accuracy:  {results['accuracy']:.4f}")
    # print(f"Precision: {results['precision']:.4f}")
    # print(f"Recall:    {results['recall']:.4f}")
    # print(f"F1:        {results['f1']:.4f}")
    # print(f"Report:\n {results['report']}\n")
    

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
perf_path = os.path.join("results", "Pipes_and_Filters_mem_sums.json")
if os.path.exists(perf_path):
    with open(perf_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
else:
    existing = []

existing.append(summary)
with open(perf_path, "w", encoding="utf-8") as f:
    json.dump(existing, f, indent=4)
