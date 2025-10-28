import joblib
import numpy as np
from Pipes_And_Filters.Filters import featureExtraction
from Pipes_And_Filters.Filters.controller import Controller
from Pipes_And_Filters.Filters.dataload.mindMonitorLoad import MindMonitorLoad
from Pipes_And_Filters.Filters.featureExtraction.featureExtract import FeatureExtract
from Pipes_And_Filters.Filters.preprocess.bandpass import Bandpass
from Pipes_And_Filters.Filters.preprocess.downsample import Downsample
from Pipes_And_Filters.Filters.preprocess.ica import Ica
from Pipes_And_Filters.Filters.preprocess.notchFilter import NotchFilter
from Pipes_And_Filters.Filters.segment.segment import Segment
from Pipes_And_Filters.Filters.segment.segmentInference import SegmentInference
from Pipes_And_Filters.data import Data

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pipe = Controller([
    (MindMonitorLoad(), {}),
    (NotchFilter(), {}),
    (Bandpass(), {"lfreq": 1.5, "hfreq": 45.0}),
    (Downsample(downsamplerate=128), {}),
    (Ica(), {"icaMethod": "fastica", "nComponents": 0.999999}),
    (SegmentInference(),{}),  
    (FeatureExtract(), {"samplingRate": 128})
])

trial_path = "Pipes_And_Filters/dataset/your_lie.csv"
packet = Data(filepath=trial_path)

packet = pipe.run(packet)

X = packet.features  

features_scaled = scaler.transform(packet.features)
predictions = model.predict(X)

mean_pred = np.mean(predictions)
print(f"Predictions per window: {predictions}")
if mean_pred > 0.5:
    print(f"Positive emotion detected.")
else:
    print(f"Negative emotion detected.")
print(f"Mean prediction over all windows: {mean_pred}")