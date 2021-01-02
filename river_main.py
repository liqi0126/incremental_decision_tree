from river import synth
from river import evaluate
from river import metrics
from river import tree

def main():
    # gen = synth.ConceptDriftStream(stream=synth.SEA(seed=42, variant=0),
    #     drift_stream = synth.SEA(seed=42, variant=1), seed = 1, position = 500, width = 50)
    gen = synth.Agrawal(classification_function=0, seed=42)
    # Take 1000 instances from the infinite data generator
    dataset = iter(gen.take(5000))

    VFDT = tree.HoeffdingAdaptiveTreeClassifier(
        grace_period=100,
        split_confidence=1e-5,
        leaf_prediction='nb',
        nb_threshold=10,
        seed=0)

    EFDT = tree.ExtremelyFastDecisionTreeClassifier(
    grace_period=100,
    split_confidence=1e-5,
    nominal_attributes=['elevel', 'car', 'zipcode'],
    min_samples_reevaluate=100
    )

    metric = metrics.Accuracy()
    result_VFDT = evaluate.progressive_val_score(dataset, VFDT, metric)
    print('VFDT')
    print(result_VFDT)
    result_EFDT = evaluate.progressive_val_score(dataset, EFDT, metric)
    print('EFDT')
    print(result_EFDT)


if __name__ == '__main__':
    main()
