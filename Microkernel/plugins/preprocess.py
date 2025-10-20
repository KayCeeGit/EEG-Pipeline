from Microkernel.plugin import Plugin
import mne

class PreprocessPlugin(Plugin):
    def __init__(self, config):
        super().__init__("Preprocess")
        self.config = config

    def execute(self, data):
        rawCopy = data.get("raw").copy()
    
        rawCopy.notch_filter(freqs=[50]) 
        
        #bandpass
        if self.config.get("bandpass", False):
            rawCopy = rawCopy.filter(l_freq=1.5, h_freq=45)

        #Downsample
        if self.config.get("downsample", False):
            rawCopy = rawCopy.resample(sfreq=int(data.get("raw").info['sfreq'] ) )

        #ICA:
        if( self.config.get("ica", False)):
            self.ica = mne.preprocessing.ICA(method= self.config.get("icaMethod"),n_components=self.config.get("n_components"), random_state=self.config.get("randSeed"), max_iter='auto')
            self.ica.fit(rawCopy)
            eogChs = [ch for ch in ['Fp1','Fp2'] if ch in rawCopy.ch_names] 
            
            if eogChs:
                eogArtifacts, _ = self.ica.find_bads_eog(rawCopy, ch_name=eogChs)
            else:
                eogArtifacts = []  
                
            muscleArtifacts, _ = self.ica.find_bads_muscle(rawCopy)

            exclude = (eogArtifacts or []) + (muscleArtifacts or [])
            exclude = [i for i in exclude if i < self.ica.n_components_]  
            self.ica.exclude = exclude
            
            rawCopy = self.ica.apply(rawCopy.copy())
            
        data["raw"] = rawCopy
        print(f"---Preprocessing Complete---")
        return data
