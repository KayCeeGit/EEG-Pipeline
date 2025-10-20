from ..filter import Filter

class Bandpass(Filter):
    def process(self,packet, lfreq=1.5, hfreq=45.0):
        rawCopy = packet.raw.copy()
        rawCopy = rawCopy.filter(l_freq=lfreq, h_freq=hfreq)
        print(f"---Bandpass Filter Applied: {lfreq}Hz - {hfreq}Hz---\n")
        packet.raw = rawCopy
        return packet