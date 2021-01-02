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
            # hack: dict required for river_main
            # new_x = {}
            # for i in range(len(x)):
            #     new_x[f'feat{i}'] = x[i]
            # x = new_x
            count += 1

            if count % self.freq == 0:
                print("Instance #%d : " % count, end="")

            for step, (learner, evaluator, performance) in enumerate(zip(self.learners, self.evaluators, self.performances)):
                predict = learner.predict_one(x)
                evaluator.add(int(y == predict))
                learner.learn_one(x, y, self.learner_metric)
                if count % self.freq == 0:
                    accuracy = evaluator.performance()
                    # error rate
                    performance.append(1 - accuracy)
                    # DEBUG
                    # if count == 181000:
                    #     print(18100)
                    #     with open('181000.txt', 'w') as f:
                    #         f.write(learner.print())
                    # if count == 182000:
                    #     print(18200)
                    #     with open('182000.txt', 'w') as f:
                    #         f.write(learner.print())
                    #     return self.performances
                    # if len(performance) > 2 and performance[-1] > performance[-2] + 0.1:
                    #     return self.performances
                    print(accuracy, end=" ")
            
            if count % self.freq == 0:
                print()
                if self.output_func is not None:
                    self.output_func(self.performances)

        return self.performances
