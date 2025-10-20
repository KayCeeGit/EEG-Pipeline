from Microkernel.plugin import Plugin
import mne
import numpy as np

class SegmentationPlugin(Plugin):
    def __init__(self, config):
        super().__init__("Segmentation")
        self.config = config #holds window size, overlap

    def execute(self, data):
        raw = data.get("raw")
        events = data.get("events")
        eventId = data.get("eventId")
        labels = data.get("labels")

        tmin, tmax = 0, 60 #trial is from 0 to 60 seconds
        
        xSegments, ySegments = [], []
        sfreq = raw.info['sfreq']
        winSample = int(self.config["windowSize"] * sfreq) 
        stepSample = int(winSample * (1 - self.config["overlap"])) 
        
        epochs = mne.Epochs(raw, events, event_id={'video': eventId}, tmin=tmin, tmax=tmax,
                            baseline=None, preload=True)
        
        epochs = epochs.drop_channels(["Status"])
        epochData = epochs.get_data()   # shape (n_trials, n_channels, n_samples)

        for trialIdx, trialData in enumerate(epochData): #going through every trial 
            label = labels[trialIdx]
            samples = trialData.shape[-1]
            start = int(3 * sfreq)  # skip first 3s for baseline
            end = samples

            # sliding windows
            for s in range(start, end - winSample + 1, stepSample):
                seg = trialData[:, s:s+winSample]  # shape (n_channels, winSample)
                xSegments.append(seg)
                ySegments.append(label)
        print(f"---Segmentation Complete---")
        data["xSegments"] = np.array(xSegments)
        data["ySegments"] = np.array(ySegments)
        return data