import mne

class Dataloader:
    def __init__(self, repo):
        self.repo = repo

    def getData(self, filepath):
        raw = self.repo.loadRaw(filepath)
        rawcopy = raw.copy().pick([ch for ch in raw.ch_names if ch.startswith((
            'Fp1', 'AF3', 'F3', 'F7', 'FC5',
            'FC1', 'C3', 'T7', 'CP5', 'CP1',
            'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
            'Fp2', 'AF4', 'Fz', 'F4', 'F8',
            'FC6', 'FC2', 'Cz', 'C4', 'T8',
            'CP6', 'CP2', 'P4', 'P8', 'PO4',
            'O2', 'Status'))] )
        events = mne.find_events(raw, stim_channel="Status", initial_event=True) #events comes as 3 columnes (sample Index, prev event code, event code), these are used for segmenting
        return rawcopy, events

    def getLabels(self, excelFile, participant):
        return self.repo.loadLabels(excelFile, participant)