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

import os
RANDOMSEED = int(os.getenv("RANDOM_SEED", 42))

class SVMModel:
    def __init__(self, C=10, kernel="linear", gamma="scale", lda=True, ldaComponents=1):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.lda = lda
        self.lda_components = ldaComponents

        self.scaler = StandardScaler()
        self.lda = LDA(n_components=ldaComponents) if lda else None
        self.svm = SVC(C=C, kernel=kernel, gamma=gamma)

    def prepareLabels(self, y):
        """
        Binarize labels: 0 if < 5, 1 if >= 5.
        This is used to convert a float valence rating to a postive/negative label
        """
        return (np.array(y) >= 5).astype(int)

    def fit(self, X, y, testSize=0.2, randomSeed=RANDOMSEED):
        # Discretize
        y_discrete = self.prepareLabels(y)

        # LDA
        if self.lda:
            X = self.lda.fit_transform(X, y_discrete)

        # Shuffle and split
        X, y_discrete = shuffle(X, y_discrete, random_state=randomSeed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_discrete, test_size=testSize, random_state=randomSeed
        )

        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Train 
        self.svm.fit(X_train, y_train)

        # Infer
        y_pred = self.svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        precision = precision_score(y_test, y_pred)


        return {"accuracy": acc, "recall": recall, "f1": f1, "report": report, "y_true": y_test, "y_pred": y_pred, "precision": precision}

    def predict(self, X):
        if self.use_lda:
            X = self.lda.transform(X)
        X = self.scaler.transform(X)
        return self.svm.predict(X)
    
class KNNModel:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    def prepareLabels(self, y):
        return (np.array(y) >= 5).astype(int)

    def fit(self, X, y, testSize=0.2, randomSeed=RANDOMSEED):
        y_discrete = self.prepareLabels(y)

        # Shuffle and split
        X, y_discrete = shuffle(X, y_discrete, random_state=randomSeed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_discrete, test_size=testSize, random_state=randomSeed
        )

        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Train 
        self.knn.fit(X_train, y_train)

        # Infer
        y_pred = self.knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred)

        return {"accuracy": acc, "recall": recall, "f1": f1, "report": report, "y_true": y_test, "y_pred": y_pred, "precision": precision, "scaler": self.scaler}

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.knn.predict(X)
    
class RandomForestModel:
    def __init__(self, nEstimators=100, maxDepth=None, random_state=RANDOMSEED):
        self.scaler = StandardScaler()
        self.rf = RandomForestClassifier(
            n_estimators=nEstimators,
            max_depth=maxDepth,
            random_state=random_state,
            n_jobs=-1
        )

    def prepareLabels(self, y):
        return (np.array(y) >= 5).astype(int)

    def fit(self, X, y, testSize=0.2, randomSeed=RANDOMSEED):
        y_discrete = self.prepareLabels(y)

        # Shuffle and split
        X, y_discrete = shuffle(X, y_discrete, random_state=randomSeed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_discrete, test_size=testSize, random_state=randomSeed
        )

        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Train 
        self.rf.fit(X_train, y_train)

        # Evaluate
        y_pred = self.rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred)

        return {"accuracy": acc, "recall": recall, "f1": f1, "report": report, "y_true": y_test, "y_pred": y_pred, "precision": precision}

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.rf.predict(X)

class MLPModel:
    def __init__(self, hiddenLayerSizes=(128, 64), activation="relu", maxIter=300,
                 lda=True, ldaComponents=1):
        self.hiddenLayerSizes = hiddenLayerSizes
        self.activation = activation
        self.maxIter = maxIter
        self.lda = lda
        self.lda_components = ldaComponents

        self.scaler = StandardScaler()
        self.lda_model = LDA(n_components=ldaComponents) if lda else None

        self.mlp = MLPClassifier(hidden_layer_sizes=hiddenLayerSizes,
                                 activation=activation,
                                 solver='adam',
                                 max_iter=maxIter,
                                 random_state=RANDOMSEED)

    def prepareLabels(self, y):
        return (np.array(y) >= 5).astype(int)

    def fit(self, X, y, testSize=0.2, randomSeed=RANDOMSEED):
        y_discrete = self.prepareLabels(y)

        if self.lda:
            X = self.lda_model.fit_transform(X, y_discrete)

        X, y_discrete = shuffle(X, y_discrete, random_state=randomSeed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_discrete, test_size=testSize, random_state=randomSeed
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.mlp.fit(X_train, y_train)

        y_pred = self.mlp.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred)

        return {"accuracy": acc, "recall": recall, "f1": f1, "report": report, "y_true": y_test, "y_pred": y_pred, "precision": precision}

    def predict(self, X):
        if self.lda:
            X = self.lda_model.transform(X)
        X = self.scaler.transform(X)
        return self.mlp.predict(X)