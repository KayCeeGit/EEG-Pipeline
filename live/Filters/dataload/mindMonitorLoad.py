import pandas as pd
import numpy as np
import mne
from ..filter import Filter


class MindMonitorLoad(Filter):
    def process(self, packet):
        df = pd.read_csv(packet.filepath)
        df.infer_objects(copy=False) #convert columns to appropriate dtypes (needed for interpolating due to some depreciation of the function)
        df.interpolate(method='linear', axis=0, limit_direction='both', inplace=True) #Muse seems to drop a row every so often, this fills that up
        ch_names = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]

        data = df[ch_names].to_numpy().T 
        info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types="eeg")
        raw = mne.io.RawArray(data, info)
        raw.rename_channels (lambda x: x.replace("RAW_", "")) #remove RAW_ prefix
        raw.set_montage("standard_1020") #^^^ so that the montage is set correctly 

        packet.raw = raw
        packet.events = None  
        print("---MindMonitor CSV loaded successfully---\n")
        return packet
