import mne
import pandas as pd

class FileRepository:
    def loadRaw(self, filepath):
        return mne.io.read_raw_bdf(filepath, preload=True)

    def loadLabels(self, participantInfoExcel, participant):
        df = pd.read_excel(participantInfoExcel)
        participant_rows = df.loc[df['Participant_id'] == participant, ['Trial', 'Valence']]
        if participant_rows.empty:
            raise ValueError(f"No labels found for participant {participant}")
        participant_rows = participant_rows.sort_values('Trial')
        return participant_rows['Valence'].to_numpy()
