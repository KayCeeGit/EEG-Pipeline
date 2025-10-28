class Data:
    """
    Holds all shared EEG processing data for the pipeline.
    filters transform/modify some part/s of this packet and returns it.
    """
    def __init__(self, filepath=None, excelFile=None, participant=None, trial = None):  
        self.raw = None          # MNE Raw object
        self.events = None       # numpy array
        self.labels = None       # array
        # self.participant = participant  # int
        self.trial = trial # int
        self.features = None     # feature matrix
        self.filepath = filepath     # path to EEG data file
        self.excelFile = excelFile   # path to labels file
        # self.xSegments = None      # segments of EEG data
        self.ySegments = None  # labels for each segment
