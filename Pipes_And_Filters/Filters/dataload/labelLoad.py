import pandas as pd
from ..filter import Filter

class LabelLoad(Filter):

    def process(self,packet):
        
        df = pd.read_excel(packet.excelFile)
        participant_rows = df.loc[df['Participant_id'] == packet.participant, ['Trial', 'Valence']]
        participant_rows = participant_rows.sort_values('Trial')
        print("---Labels Loaded---\n")
        packet.labels = participant_rows['Valence'].to_numpy()
        packet.participant = packet.participant
        return packet

        
