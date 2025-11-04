import numpy as np
from pylsl import StreamInlet, resolve_streams
from scipy.fft import rfft, rfftfreq
import joblib

SFREQ = 256          
WINDOW_SIZE = 4.0   #these need to match the segment settings
OVERLAP = 0.5   #see above
BANDS = {   #same as feature extraction 
    "delta": (0.5, 4),
    "theta": (4, 7),
    "alpha": (7, 12),
    "beta":  (12, 30),
    "gamma": (30, 45),
}

MODEL_PATH = "live/model.pkl"
SCALER_PATH = "live/scaler.pkl"

print("Loading trained model")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Model Loaded\n")

print("looking for Muse data")
streams = resolve_streams(5)
stream = next((s for s in streams if s.type() == "EEG"), None)
if stream is None:
    raise RuntimeError("stream is not found")

inlet = StreamInlet(stream)
print(f"Connected to stream: {stream.name()}")

winSamples = int(WINDOW_SIZE * SFREQ)
stepSamples = int(winSamples * (1 - OVERLAP))
buffer = []

def fft(seg, sfreq): 
    seg = seg[:4, :] 
    yf = np.abs(rfft(seg, axis=-1)) ** 2
    xf = rfftfreq(seg.shape[-1], 1 / sfreq)
    features = [np.sum(yf[:, (xf >= low) & (xf <= high)], axis=-1)
                for low, high in BANDS.values()]
    return np.concatenate(features, axis=-1)

print("Running Infernence")

try:
    while True:
        sample, _ = inlet.pull_sample()
        buffer.append(sample)

        if len(buffer) >= winSamples:
            window = np.array(buffer[-winSamples:]).T
            features = fft(window, SFREQ)
            feats_scaled = scaler.transform(features.reshape(1, -1))
            pred = model.predict(feats_scaled)[0]
            if pred == 1:
                emotion = "happy"
            elif pred == 0:
                emotion = "sad"
            else:
                emotion = "No Prediction Found"
            print(f"Predicted emotion: {emotion}")
            
            buffer = buffer[stepSamples:] 


except KeyboardInterrupt:
    print("\n Stopped")
