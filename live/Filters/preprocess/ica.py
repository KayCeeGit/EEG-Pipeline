from ..filter import Filter
import mne

class Ica(Filter):
    def __init__(self, icaMethod = "fastica", randSeed= 42, nComponents=0.999999):
        self.randSeed = randSeed
        self.icaMethod = icaMethod
        self.nComponents = nComponents
    
    def process(self, packet):

        rawCopy = packet.raw.copy()
        
        ica = mne.preprocessing.ICA(method=self.icaMethod,n_components=self.nComponents, random_state=self.randSeed, max_iter='auto')
        ica.fit(rawCopy)
        eogChs = [ch for ch in ['Fp1','Fp2'] if ch in rawCopy.ch_names] #using the frontal channels as EOG reference - Fp1 and Fp2 are the only frontal channels in DEAP
        
        if eogChs:
            eogArtifacts, _ = ica.find_bads_eog(rawCopy, ch_name=eogChs)
        else:
            eogArtifacts = []  
            
        muscleArtifacts, _ = ica.find_bads_muscle(rawCopy)
        
        exclude = (eogArtifacts or []) + (muscleArtifacts or [])
        exclude = [i for i in exclude if i < ica.n_components_]  #remove any eog or muscle components
        ica.exclude = exclude
        
        print("---ICA Applying---\n")
        rawCopy = ica.apply(rawCopy.copy())
        print("---ICA Applied---\n")
        packet.raw = rawCopy
        return packet