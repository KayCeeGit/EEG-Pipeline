import argparse
import json
import joblib
import numpy as np
from tqdm import tqdm
import time
import psutil
import os
import tracemalloc

from Pipes_And_Filters.Filters.dataload.eegDataLoad import EegDataLoad
from Pipes_And_Filters.Filters.dataload.labelLoad import LabelLoad
from Pipes_And_Filters.Filters.dataload.mindMonitorLoad import MindMonitorLoad

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
    (MindMonitorLoad(), {}),
    # (EegDataLoad(), {}),
    (LabelLoad(), {"excelFile": "participant_ratings.xls", "participant": 1}),
    (NotchFilter(), {}),
    (Bandpass(), {"lfreq": 1.5, "hfreq": 45.0}),
    (Downsample(downsamplerate=128), {}),
    (Ica(), {"icaMethod": "fastica", "randSeed": RANDOMSEED, "nComponents": 0.999999}),
    (Segment(eventID= 4),{}),  
    (FeatureExtract(), {"samplingRate": 128}),
]

models = [
    # ("SVMlda-0.1", SVMModel(C=0.1, kernel="linear", lda=True)),
    # ("SVMlda-10", SVMModel(C=10, kernel="linear", lda=True)),
    # ("SVMnoLda-0.1", SVMModel(C=0.1, kernel="linear", lda=False)),
    # ("SVMnoLda-10", SVMModel(C=10, kernel="linear", lda=False)),
    # ("KNN-5", KNNModel(n_neighbors=5)),
    ("KNN-7", KNNModel(n_neighbors=7)),
    # ("KNN-25", KNNModel(n_neighbors=25)),
    # ("RF-100", RandomForestModel(nEstimators=100)),
    # ("RF-500", RandomForestModel(nEstimators=500)),
    # ("MLPrelu", MLPModel(activation="relu")),
    # ("MLPtanh", MLPModel(activation="tanh")),
    # ("MLPlogistic", MLPModel(activation="logistic")),
]


pipe = Controller(filters)
allX = []
allY = []

for i in tqdm(range(1, 11)):
    filepath = f"Pipes_And_Filters\dataset\\trial{i:02d}.csv"
    packet = Data(filepath=filepath, excelFile="Pipes_And_Filters\dataset\\datasetLabels.xlsx", trial=i)
    output = pipe.run(packet)
    allX.append(output.features)
    allY.append(output.ySegments)
    
allX = np.vstack(allX)
allY = np.hstack(allY)
print(f"AllX = {allX.shape}, allY = {allY.shape}")


for name, model in models:
    print(f"\n===== Running {name} =====")
    results = model.fit(allX, allY)
    joblib.dump(model, "model.pkl")
    joblib.dump(results["scaler"], "scaler.pkl")
    # log = {
    #         "architecture": "PipesAndFilters",
    #         "model": name,
    #         "accuracy": results["accuracy"],
    #         "precision": results["precision"],
    #         "recall": results["recall"],
    #         "f1": results["f1"],
    #         "report": results["report"],
    #     }

    # os.makedirs("results", exist_ok=True)
    # logPath = os.path.join("results", f"Pipes_And_Filters_model_results.json")

    # # Append instead of overwriting
    # if os.path.exists(logPath):
    #     with open(logPath, "r") as f:
    #         data = json.load(f)
    # else:
    #     data = []

    # data.append(log)
    # with open(logPath, "w") as f:
    #     json.dump(data, f, indent=4)
    # print(f"Accuracy:  {results['accuracy']:.4f}")
    # print(f"Precision: {results['precision']:.4f}")
    # print(f"Recall:    {results['recall']:.4f}")
    # print(f"F1:        {results['f1']:.4f}")
    # print(f"Report:\n {results['report']}\n")

