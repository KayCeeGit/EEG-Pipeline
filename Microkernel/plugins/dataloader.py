from Microkernel.plugin import Plugin
import mne

class DataLoadPlugin(Plugin):
    def __init__(self):
        super().__init__("DataLoader")

    # def getName(self):
    #     return "EEG Data Loader"

    def execute(self, data):

        filepath = data.get("filepath") 
        
        raw = mne.io.read_raw_bdf(filepath, preload=True)
        rawcopy = raw.copy().pick([ch for ch in raw.ch_names if ch.startswith((
            'Fp1', 'AF3', 'F3', 'F7', 'FC5',
            'FC1', 'C3', 'T7', 'CP5', 'CP1',
            'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
            'Fp2', 'AF4', 'Fz', 'F4', 'F8',
            'FC6', 'FC2', 'Cz', 'C4', 'T8',
            'CP6', 'CP2', 'P4', 'P8', 'PO4',
            'O2', 'Status'))] )

        events = mne.find_events(rawcopy, stim_channel="Status", initial_event=True)
        data["raw"] = rawcopy
        data["events"] = events
        print(f"---Data Loaded---")
        return data
