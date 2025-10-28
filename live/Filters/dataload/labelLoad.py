import pandas as pd
from ..filter import Filter

class LabelLoad(Filter):

    def process(self,packet):
        df = pd.read_excel(packet.excelFile)
        trialRows = df.loc[df['Trial'] == packet.trial, ['Trial', 'Valence']]
        trialRows = trialRows.sort_values('Trial')
        print("---Labels Loaded---\n")
        packet.labels = trialRows['Valence'].to_numpy()
        # packet.participant = packet.participant Why did i put this here?
        return packet

        
