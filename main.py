from dataset.stream import DataStream
import dataset.config
from evaluation.EvaluatePrequential import EvaluatePrequential

from metrics.gini import gini

from model.utils import AttrType
from model.vfdt import VfdtTree
from model.efdt import EfdtTree

import time
import datetime
import argparse
import pickle
from matplotlib import pyplot as plt
import numpy as np
import random
from copy import deepcopy

from river import tree

def arg_parse():
    parser = argparse.ArgumentParser(description='Incremental Decision Tree')
    parser.add_argument("--seed", type=int, default=4096)
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument('--tree', nargs='+', type=str,
                        choices=['v', 'e', 'vfdt', 'efdt', 'river-v', 'river-e'], default=['v', 'e'])
    parser.add_argument('--dataset', type=str, default='forest')
    parser.add_argument('--max_instance', type=int, default=200000)
    parser.add_argument('--exp', type=str, default=datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d_%H:%M:%S'))
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()

    args = arg_parse()
    args.exp = args.dataset + '_' + args.exp
    if args.shuffle:
        args.exp += '_shuffle'
    args.exp += f'_seed{args.seed}'

    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dataset == 'poker':
        attrTypes = [AttrType.CATE] * 10
    elif args.dataset.startswith('moa'):
        attrTypes = [AttrType.NUME] * 5
        # attrTypes = [AttrType.CATE] * 5
    # elif args.dataset == 'forest':
    #     attrTypes = [AttrType.CATE] * 54
    else:
        attrTypes = None
    stream = DataStream(dataset.config.csv_path[args.dataset], attrTypes=attrTypes, shuffle=args.shuffle, seed=args.seed)
    candidate_attr, n_class = stream.attributes, stream.n_class

    for attr in candidate_attr:
        attr.print()

    # hyperparameter
    delta = 0.01
    max_depth = 100
    tau = 0.05

    def metric_func(class_freq):
        return -gini(np.fromiter(class_freq.values(), dtype=int))
    # def metric_func(class_freq): return -gini(class_freq)

    models = []
    legend = []
    if 'v' in args.tree or 'vfdt' in args.tree:
        models.append(VfdtTree)
        legend.append('VFDT')
    if 'e' in args.tree or 'efdt' in args.tree:
        models.append(EfdtTree)
        legend.append('EFDT')
    if 'river-v' in args.tree:
        models.append(tree.HoeffdingAdaptiveTreeClassifier)
        legend.append('RIVER-VFDT')
    if 'river-e' in args.tree:
        models.append(tree.ExtremelyFastDecisionTreeClassifier)
        legend.append('RIVER-EFDT')

    learners = []
    for i, model in enumerate(models):
        learners.append(model(candidate_attr=deepcopy(candidate_attr),
                              n_class=n_class,
                            #   delta=delta,
                              max_depth=max_depth,
                              tau=tau))

    def output(performances):
        output_path = 'outputs/%s.pickle' % args.exp
        with open(output_path, 'wb') as f:
            pickle.dump({
                'dataset': args.dataset,
                'shuffle': stream.shuffle,
                'seed': stream.seed,
                'delta': delta,
                'max_depth': max_depth,
                'tau': tau,
                'learners': legend,
                'performances': performances,
            }, f)

    eval = EvaluatePrequential(
        stream, learners, metric_func, max_inst=args.max_instance, output_func=output)
    performances = eval.doMainTask()

    print('Total time: ', time.time() - start_time)

    for line in performances:
        plt.plot(line)
    
    for learner in learners:
        learner.print()

    plt.title("%s dataset" % args.dataset)
    plt.xlabel("instances \\times 1000")
    plt.ylabel("error rate")
    plt.legend(labels=legend)
    plt.show()

