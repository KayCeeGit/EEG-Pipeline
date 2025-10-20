from Microkernel.plugin import Plugin
import pandas as pd

class LabelLoadPlugin(Plugin):
    def __init__(self):
        super().__init__("LabelLoader")

    # def get_name(self):
    #     return "Label Loadr"

    def execute(self, data):
        """
        gets the label data for the specified participant
        """
        participantInfoExcel = data.get("excelFile")
        participant = data.get("participant")
        df = pd.read_excel(participantInfoExcel)
        
        participantRows = df.loc[df['Participant_id'] == participant, ['Trial', 'Valence']]
        participantRows = participantRows.sort_values('Trial')

        data["labels"] = participantRows["Valence"].to_numpy()
        print(f"---Labels Loaded---")
        return data
