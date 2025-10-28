from ..filter import Filter
import mne
import numpy as np

class SegmentInference(Filter):
    
    def __init__(self, windowSize=4.0, overlap=0.5):
        self.windowSize = windowSize
        self.overlap = overlap
        # self.eventId = eventID
    
    
    def process(self, packet):
        raw = packet.raw
        data = raw.get_data() 
        sfreq = packet.raw.info['sfreq']

        winSample = int(self.windowSize * sfreq)
        stepSample = int(winSample * (1 - self.overlap))
        xSegments, ySegments = [], []

        samples = data.shape[-1]
        start = int(3 * sfreq)  #skip first 3s
        end = samples


        for s in range(start, end - winSample + 1, stepSample):
            seg = data[:, s:s + winSample]  
            xSegments.append(seg)

        print(f"---Segments Extracted {len(xSegments)}---\n")
        packet.xSegments = np.array(xSegments)
        return packet