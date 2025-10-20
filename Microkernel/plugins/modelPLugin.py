import os
from Microkernel.plugin import Plugin
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

RANDOMSEED = int(os.getenv("RANDOM_SEED", 42))

class ModelPlugin(Plugin):
    def __init__(self, config=None, random_seed=None):
        super().__init__("Models")
        self.config = config or {}
        if random_seed is not None:
            self.config["randomSeed"] = random_seed
        self.scaler = StandardScaler()

    def execute(self, data):
        X = data.get("features")
        y = data.get("labels")
        y_discrete = (np.array(y) >= 5).astype(int)

        modelType = self.config.get("modelType")

        if self.config.get("lda", False):
            lda = LDA(n_components=self.config.get("ldaComponents", 1))
            X = lda.fit_transform(X, y_discrete)

        X, y_discrete = shuffle(X, y_discrete, random_state=self.config.get("randomSeed", RANDOMSEED))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_discrete, test_size=self.config.get("testSize", 0.2),
            random_state=self.config.get("randomSeed", RANDOMSEED)
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        if modelType == "SVM":
            model = SVC(C=self.config.get("C", 1.0), kernel=self.config.get("kernel", "linear"))
        elif modelType == "KNN":
            model = KNeighborsClassifier(n_neighbors=self.config.get("n_neighbors", 5))
        elif modelType == "RF":
            model = RandomForestClassifier(n_estimators=self.config.get("n_estimators", 100), random_state=RANDOMSEED)
        elif modelType == "MLP":
            model = MLPClassifier(
                hidden_layer_sizes=self.config.get("hiddenLayerSizes", (128, 64)),
                activation=self.config.get("activation", "relu"),
                max_iter=self.config.get("maxIter", 300),
                random_state=RANDOMSEED
            )
        else:
            raise ValueError(f"Unknown model type: {modelType}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        report = classification_report(y_test, y_pred, zero_division=0)

        # print(f"--- {modelType} Results ---")
        # print(f"Accuracy:  {acc:.4f}")
        # print(f"Precision: {precision:.4f}")
        # print(f"Recall:    {recall:.4f}")
        # print(f"F1:        {f1:.4f}")
        # print(report)
        
        return {
            "modelType": modelType,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "report": report
        }
