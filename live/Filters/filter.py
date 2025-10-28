from abc import ABC, abstractmethod

class Filter(ABC):
    "Pipes and filters abstract class"
    
    @abstractmethod
    def process(self, data):
        "Where filters manipulate or change the data before passing to a pipe"
        pass