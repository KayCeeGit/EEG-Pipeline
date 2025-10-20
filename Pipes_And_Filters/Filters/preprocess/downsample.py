from ..filter import Filter

class Downsample(Filter):
    def __init__(self, downsamplerate):
        self.downsamplerate = downsamplerate
        
    def process(self, packet):
        rawCopy = packet.raw.copy()
        rawCopy = rawCopy.resample(sfreq=int(self.downsamplerate))
        print(f"---Downsampled to {self.downsamplerate}Hz---\n")
        packet.raw = rawCopy
        return packet