import numpy as np
import mne

class Segmentation:
    def __init__(self, windowSize=4.0, overlap=0.75):
        """
        Initialises class
        parameters:
        windowSize (int) : length of each epoch in seconds. Default is 4 seconds.
        overlap (float): fraction overlap between windows (0.0-0.9) Deafault is 0.75 as it achieved best results in testing.
        """
        self.windowSize = windowSize
        self.overlap = overlap
        
    def getEpochs(self, raw, events, eventId, labels):
        """
        Segments the raw data into epochs
        
        parameters:
        raw (mne.io.Raw): preprocessed MNE Raw from Preprocessing.preprocess()
        events (np.array): event markers from Dataloader.getData()
        eventId (int): event code to provide marker of dataset eeg collection start (e.g. {4: 'video'} for DEAP data)
        labels (np.array): np.array of labels from Dataloader.getLabels(), same length as number of trials in events
        
        returns:
        X_segments : (n_epochs, n_channels, n_samples) - segmented EEG data
        y_labels : (n_epochs) - labels for each epoch
        """
        sfreq = raw.info['sfreq']
        winSample = int(self.windowSize * sfreq) 
        stepSample = int(winSample * (1 - self.overlap)) 

        tmin, tmax = 0, 60 #trial is from 0 to 60 seconds
        epochs = mne.Epochs(raw, events, event_id={'video': eventId}, tmin=tmin, tmax=tmax,
                            baseline=None, preload=True)
        epochs = epochs.drop_channels(["Status"])
        data = epochs.get_data()   # shape (n_trials, n_channels, n_samples)

        X_segments, y_segments = [], []

        for trialIdx, trialData in enumerate(data): 
            label = labels[trialIdx]
            n_samples = trialData.shape[-1]
            start = int(3 * sfreq)  # skip first 3s for baseline
            end = n_samples

            # sliding windows
            for s in range(start, end - winSample + 1, stepSample):
                seg = trialData[:, s:s+winSample]  # shape (n_channels, winSample)
                X_segments.append(seg)
                y_segments.append(label)

        return np.array(X_segments), np.array(y_segments)