from ..data import Data

class Controller:
    def __init__(self, filters,params=None):
        self.filters = []  
        self.filters = filters
        self.parameters = params or {}


    # def add_filter(self, filters):
    #      self.filters.append(filter)  

    def run(self,packet):
        """Pushes data sequentially through each filter."""
        for filter, params in self.filters:
            print(f"--- Running {filter.__class__.__name__} ---")
            params = self.parameters.get(filter.__class__.__name__, {})
            packet = filter.process(packet, **params)
 
        return packet 