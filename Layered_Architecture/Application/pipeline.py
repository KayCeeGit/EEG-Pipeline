import numpy as np
from tqdm import tqdm

class Pipeline:
    def __init__(self, dataloader, preprocessor, segmenter, featureExtractor, model=None):
        self.dataloader = dataloader
        self.preprocessor = preprocessor
        self.segmenter = segmenter
        self.featureExtractor = featureExtractor
        self.model = model  # optional — can be swapped later

    def run(self, fileUpload, labelUpload, participant):
        """Run the full pipeline for a single participant."""
        raw, events = self.dataloader.getData(fileUpload)
        labels = self.dataloader.getLabels(labelUpload, participant) 
        praw = self.preprocessor.preprocess(raw, True, 128, True, True) #Lowering the sample rate below 128Hz will have issues (See Nyquist Law)
        X_segments, y_segments = self.segmenter.getEpochs(praw, events, 4, labels)
        X, y = self.featureExtractor.extract(X_segments, y_segments, praw.info['sfreq'])
        return X, y, [participant] * len(y)

    def prepareAll(self, labelUpload, participantsRange):
        """Aggregate processed data across multiple participants."""
        allX, allYLabels, participants = [], [], []

        for i in tqdm(participantsRange):
            fileUpload = f"s{i:02d}.bdf"
            X, y, p_ids = self.run(fileUpload, labelUpload, i)
            allX.append(X)
            allYLabels.append(y)
            participants.extend(p_ids)

        # stack into arrays
        allX = np.vstack(allX)
        allYLabels = np.hstack(allYLabels)
        participants = np.array(participants)

        return allX, allYLabels, participants

    def evaluateModel(self, X, y, model):
        """Train + evaluate a specific model on prepared data."""
        results = model.fit(X, y)
        print("\nModel results:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Recall:   {results['recall']:.4f}")
        print(f"F1:       {results['f1']:.4f}")
        print(results['report'])
        return results
