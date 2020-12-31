from .WindowClassificationPerformanceEvaluator import *


class EvaluatePrequential:
    def __init__(self, stream, learner, learner_metric, evaluator=WindowClassificationPerformanceEvaluator(), freq=1000, max_inst=2000000):
        self.stream = stream
        self.learner = learner
        self.learner_metric = learner_metric
        self.evaluator = evaluator

        self.freq = freq
        self.max_inst = max_inst

    def doMainTask(self):
        count = 0
        self.performance = []
        while count < self.max_inst:
            instance = self.stream.nextInstance()
            if instance is None:
                break
            x, y = instance
            predict = self.learner._predict(x)
            self.evaluator.add(int(y == predict))
            self.learner._update(x, y, self.learner_metric)

            count += 1
            if count % self.freq == 0:
                accuracy = self.evaluator.performance()
                # error rate
                self.performance.append(1 - accuracy)
                print(count, accuracy)

        return self.performance
