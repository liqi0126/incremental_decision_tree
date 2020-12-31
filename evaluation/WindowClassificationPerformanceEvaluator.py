
class WindowClassificationPerformanceEvaluator:
    def __init__(self, window=1000):
        self.window = window
        self.results = []
    
    def add(self, result):
        self.results.append(result)
        if (len(self.results) > self.window):
            self.results.pop(0)

    def performance(self):
        if len(self.results) == 0:
            return None
        else:
            return sum(self.results) / len(self.results)