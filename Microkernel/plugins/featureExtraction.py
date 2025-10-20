from Microkernel.plugin import Plugin
import numpy as np
from scipy.fft import rfft, rfftfreq

class FeatureExtractionPlugin(Plugin):
    def __init__(self):
        super().__init__("FeatureExtraction")
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 7),
            'alpha': (7, 12),
            'beta':  (12, 30),
            'gamma': (30, 45),
        }

    def fftBandPowers(self, seg, samplerate):

        signalStrengthYF = np.abs(rfft(seg, axis=-1)) ** 2
        frequecnyBinXF = rfftfreq(seg.shape[-1], 1 / samplerate)
        
        features = [np.sum(signalStrengthYF[:, (frequecnyBinXF >= low) & (frequecnyBinXF <= high)], axis=-1)
                    for low, high in self.bands.values()]
        features = np.array(features)  
        features = np.log(features + 1e-8)

        return np.concatenate(features, axis=-1) 
        
    def execute(self, data):
        raw = data.get("raw")
        xSegments = data.get("xSegments")
        ySegments = data.get("ySegments")
        
        X = [self.fftBandPowers(seg, raw.info["sfreq"]) for seg in xSegments]
        X = np.stack(X, axis=0)
        
        data["features"] = X
        data["labels"] = ySegments
        print("---Features Extracted---\n")
        return data