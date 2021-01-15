from .WindowClassificationPerformanceEvaluator import *
from tqdm import tqdm

class EvaluatePrequential:
    def __init__(
        self, 
        stream, 
        learners, 
        learner_metric, 
        evaluator=WindowClassificationPerformanceEvaluator, 
        freq=1000, 
        max_inst=2000000, 
        output_func=None
    ):
        self.stream = stream
        self.n_learners = len(learners)
        self.learners = learners
        self.learner_metric = learner_metric
        self.evaluators = [evaluator() for _ in range(self.n_learners)]

        self.freq = freq
        self.max_inst = max_inst

        self.output_func = output_func

    def doMainTask(self, verbose=True):
        self.stream.reset()
        count = 0
        self.performances = [[] for _ in range(self.n_learners)]

        if self.output_func is not None:
            self.output_func(self.performances)

        if verbose:
            pbar = tqdm(range(self.max_inst)[::self.freq])

        while count < self.max_inst:
            instance = self.stream.nextInstance()
            if instance is None:
                break
            x, y = instance
            count += 1

            current_accuracys = []
            for step, (learner, evaluator, performance) in enumerate(zip(self.learners, self.evaluators, self.performances)):
                predict = learner.predict_one(x)
                evaluator.add(int(y == predict))
                learner.learn_one(x, y, self.learner_metric)
                if count % self.freq == 0:
                    accuracy = evaluator.performance()
                    performance.append(1 - accuracy)
                    current_accuracys.append(accuracy)
            
            if count % self.freq == 0:
                if self.output_func is not None:
                    self.output_func(self.performances)
            
            if verbose and count % self.freq == 0:
                 pbar.set_description("Instance #%d : " % count + ", Acc: " + str(current_accuracys))
                 pbar.update(1)

        return self.performances
