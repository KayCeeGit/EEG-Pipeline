class Kernel:
    def __init__(self):
        self.plugins = {}
        
    def registerPlugin(self, plugin, name=None):
        name = name or plugin.__class__.__name__
        self.plugins[name] = plugin

    def execute(self, data):
        result = data
        for plugin in self.plugins.values():
            result = plugin.execute(result)
        return result
    
    # def execute(self, data): #minor change to return the result of the model plugin for logging
    #     result = None
    #     for plugin in self.plugins:
    #         data = plugin.execute(data)
    #         if isinstance(data, dict) and "accuracy" in data:
    #             result = data
    #     return result