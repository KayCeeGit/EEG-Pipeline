from ..filter import Filter

class NotchFilter(Filter):
    def process(self, packet):
        rawCopy = packet.raw.copy()
        rawCopy.notch_filter(freqs=[50]) 
        print("---Notch Filter Applied---\n")
        packet.raw = rawCopy
        return packet