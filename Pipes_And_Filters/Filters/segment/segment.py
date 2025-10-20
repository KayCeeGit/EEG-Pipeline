from ..filter import Filter
import mne
import numpy as np

class Segment(Filter):
    
    def __init__(self, windowSize=4.0, overlap=0, eventID=4):
        self.windowSize = windowSize
        self.overlap = overlap
        self.eventId = eventID
    
    
    def process(self, packet,):
        sfreq = packet.raw.info['sfreq']
        winSample = int(self.windowSize * sfreq) 
        stepSample = int(winSample * (1 - self.overlap)) 

        tmin, tmax = 0, 60 #trial is from 0 to 60 seconds
        epochs = mne.Epochs(packet.raw, packet.events, event_id={'video': self.eventId}, tmin=tmin, tmax=tmax,
                            baseline=None, preload=True)
        epochs = epochs.drop_channels(["Status"])
        data = epochs.get_data()   # shape (n_trials, n_channels, n_samples)

        xSegments, ySegments = [], []

        for trialIdx, trialData in enumerate(data): 
            label = packet.labels[trialIdx]
            samples = trialData.shape[-1]
            start = int(3 * sfreq)  # skip first 3s for baseline
            end = samples

            # sliding windows
            for s in range(start, end - winSample + 1, stepSample):
                seg = trialData[:, s:s+winSample]  # shape (n_channels, winSample)
                xSegments.append(seg)
                ySegments.append(label)
        print("---Segments Extracted---\n")
        packet.xSegments =  np.array(xSegments)
        packet.ySegments = np.array(ySegments)
        return packet