import mne

class Preprocessing:
    def __init__(self, icaMethod, randSeed, nComponents=0.999999):
        self.randSeed = randSeed
        self.icaMethod = icaMethod
        self.nComponents = nComponents

    def preprocess(self, raw, downsampleTrue, downsamplerate, bandpassTrue, icaTrue):
        rawCopy = raw.copy()
    
        rawCopy.notch_filter(freqs=[50]) #Include this for Muse data/ testing. Removes mains static. HOWEVER, if bandpass if selected this is redunant as Bp filter filters out >50Hz noise anyway.
        
        #bandpass
        if(bandpassTrue == True):
            rawCopy = rawCopy.filter(l_freq=1.5, h_freq=45)

        #Downsample
        if downsampleTrue:
            rawCopy = rawCopy.resample(sfreq=int(downsamplerate))

        #ICA:
        if(icaTrue == True):
            self.ica = mne.preprocessing.ICA(method=self.icaMethod,n_components=self.nComponents, random_state=self.randSeed, max_iter='auto')
            self.ica.fit(rawCopy)
            eogChs = [ch for ch in ['Fp1','Fp2'] if ch in rawCopy.ch_names] #using the frontal channels as EOG reference - Fp1 and Fp2 are the only frontal channels in DEAP
            
            if eogChs:
                eogArtifacts, _ = self.ica.find_bads_eog(rawCopy, ch_name=eogChs)
            else:
                eogArtifacts = []  
                
            muscleArtifacts, _ = self.ica.find_bads_muscle(rawCopy)
            
            exclude = (eogArtifacts or []) + (muscleArtifacts or [])
            exclude = [i for i in exclude if i < self.ica.n_components_]  #remove any eog or muscle components
            self.ica.exclude = exclude
            
            rawCopy = self.ica.apply(rawCopy.copy())


            
        return rawCopy