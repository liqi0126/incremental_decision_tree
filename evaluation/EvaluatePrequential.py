from .WindowClassificationPerformanceEvaluator import *


class EvaluatePrequential:
    def __init__(self, stream, learners, learner_metric, evaluator=WindowClassificationPerformanceEvaluator, freq=1000, max_inst=2000000, output_func=None):
        self.stream = stream
        self.n_learners = len(learners)
        self.learners = learners
        self.learner_metric = learner_metric
        self.evaluators = [evaluator() for _ in range(self.n_learners)]

        self.freq = freq
        self.max_inst = max_inst

        self.output_func = output_func

    def doMainTask(self):
        self.stream.reset()
        count = 0
        self.performances = [[] for _ in range(self.n_learners)]

        if self.output_func is not None:
            self.output_func(self.performances)

        while count < self.max_inst:
            instance = self.stream.nextInstance()
            if instance is None:
                break
            x, y = instance
            count += 1

            if count % self.freq == 0:
                print("Instance #%d : " % count, end="")

            for learner, evaluator, performance in zip(self.learners, self.evaluators, self.performances):
                predict = learner._predict(x)
                evaluator.add(int(y == predict))
                learner._update(x, y, self.learner_metric)
                if count % self.freq == 0:
                    accuracy = evaluator.performance()
                    # error rate
                    performance.append(1 - accuracy)
                    print(accuracy, end=" ")
            
            if count % self.freq == 0:
                print()
                if self.output_func is not None:
                    self.output_func(self.performances)

        return self.performances
