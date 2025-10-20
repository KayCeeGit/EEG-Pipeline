from Pipes_And_Filters.Filters.filter import Filter


class ModelFilter(Filter):
    def __init__(self, model, modelname):
        self.model = model
        self.modelname = modelname

    def process(self, packet):
        results = self.model.fit(packet.features, packet.y_segments)
        packet.model_results = results
        
        # print(f"{self.modelname} Results:")
        # print(f"Accuracy: {results['accuracy']:.4f}")
        # print(f"Recall:   {results['recall']:.4f}")
        # print(f"F1:       {results['f1']:.4f}")
        # print(f"Precision:{results['precision']:.4f}")
        # print(f"Report:\n {results['report']}\n")

        
        return packet