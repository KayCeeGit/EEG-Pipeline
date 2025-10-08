import numpy as np
from scipy.fft import rfft, rfftfreq

class FeatureExtraction:
    def __init__(self):
        """
        Initialize the FeatureExtraction class using the predefined EEG frequency bands national library of medicine: https://www.ncbi.nlm.nih.gov/books/NBK539805/
        """
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 7),
            'alpha': (7, 12),
            'beta':  (12, 30),
            'gamma': (30, 45),
        }

    def fftBandPowers(self, seg, samplingRate):
        """
        performs Fast Fourier Transform (FFT) on a single EEG segment and computes the log-transformed band powers
        returns:
        (n_channels * n_bands,) 
        """
        yf = np.abs(rfft(seg, axis=-1)) ** 2
        xf = rfftfreq(seg.shape[-1], 1 / samplingRate)

        features = [np.sum(yf[:, (xf >= low) & (xf <= high)], axis=-1)
                    for low, high in self.bands.values()]
        features = np.array(features)  
        features = np.log(features + 1e-8)

        return np.concatenate(features, axis=-1)  # (n_channels * n_bands,)

    def extract(self, segments, labels, samplingRate):
        """
        Extracts features from all segments using FFT band powers.
        returns:
        segments : (n_epochs, n_channels, n_samples) 
        labels   : (n_epochs,) 
        """
        
        X = [self.fftBandPowers(seg, samplingRate) for seg in segments]
        X = np.stack(X, axis=0)
        return X, labels