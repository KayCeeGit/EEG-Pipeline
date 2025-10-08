import sys
import os
import time
import psutil

from Infrastructure.File_repo import FileRepository
from Domain.dataloader import Dataloader
from Domain.preprocessing import Preprocessing
from Application.pipeline import Pipeline
from Domain.segmentation import Segmentation
from Domain.feature_extraction import FeatureExtraction
from Domain.models import SVMModel, KNNModel, RandomForestModel, MLPModel

from Presentation.visualization import plot_confusion

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    start_time = time.time()
    performance = psutil.Process(os.getpid())

    
    repo = FileRepository()
    dataloader = Dataloader(repo)
    preprocessor = Preprocessing(icaMethod="fastica", randSeed=42)
    segmenter = Segmentation(windowSize=4.0, overlap=0.75)
    featureExtractor = FeatureExtraction()
    pipeline = Pipeline(dataloader, preprocessor, segmenter, featureExtractor)

    cpuAtStart = performance.cpu_times()
    memAtStart = performance.memory_info().rss / (1024 * 1024)

    allX, allY, participants = pipeline.prepareAll("participant_ratings.xls", range(1, 23))


    # svmModel = SVMModel(C=10, kernel="linear", lda=True)
    # knnModel = KNNModel(n_neighbors=5)
    # rfModel = RandomForestModel(nEstimators=100, maxDepth=None)
    

    models = [
        ("SVMlda-0.1", SVMModel(C=0.1, kernel="linear", lda=True)),
        ("SVMlda-10", SVMModel(C=10, kernel="linear", lda=True)),
        # ("SVMlda-100", SVMModel(C=100, kernel="linear", lda=True)),
        # ("SVMlda-500", SVMModel(C=500, kernel="linear", lda=True)),
        # ("SVMlda-1000", SVMModel(C=1000, kernel="linear", lda=True)),

        ("SVMnoLda-0.1", SVMModel(C=0.1, kernel="linear", lda=False)),
        ("SVMnoLda-10", SVMModel(C=10, kernel="linear", lda=False)),

        ("KNN-5", KNNModel(n_neighbors=5)),
        ("KNN-7", KNNModel(n_neighbors=7)),
        ("KNN-25", KNNModel(n_neighbors=25)),
        ("Random Forest=100", RandomForestModel(nEstimators=100, maxDepth=None)),
        ("Random Forest=500", RandomForestModel(nEstimators=500, maxDepth=None)),
        # ("Random Forest=1000", RandomForestModel(nEstimators=1000, maxDepth=None)),
        ("MLPrelu", MLPModel(hiddenLayerSizes=(128, 64), activation="relu")),
        ("MLPtanh", MLPModel(hiddenLayerSizes=(128, 64), activation="tanh")),
        ("MLPlogistic", MLPModel(hiddenLayerSizes=(128, 64), activation="logistic")),        
    ]

    for name, model in models:
        print(f"\n===== Running {name} Model =====")
        results = pipeline.evaluateModel(allX, allY, model)
        # plot_confusion(results["y_true"], results["y_pred"])
    # pipeline = Pipeline(dataloader, preprocessor, segmenter, featureExtractor, svmModel)
    # pipeline.runAll("participant_ratings.xls", participantsRange=range(1, 3))
    
    
    cpuAtEnd = performance.cpu_times()
    memAtEnd = performance.memory_info().rss / (1024 * 1024)
    print("\n=== Performance Summary ===")
    print(f"CPU User Time: {cpuAtEnd.user - cpuAtStart.user:.2f} s")
    print(f"CPU System Time: {cpuAtEnd.system - cpuAtStart.system:.2f} s")
    print(f"Memory Before: {memAtStart:.2f} MB")
    print(f"Memory After: {memAtEnd:.2f} MB")
    print("--- %s seconds ---" % (time.time() - start_time))
