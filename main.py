from dataset.stream import DataStream
import dataset.config
from evaluation.EvaluatePrequential import EvaluatePrequential

from metrics.gini import gini

from model.utils import Attr, AttrType
from model.tree import ClsTree
from model.vfdt import VfdtTree
from model.efdt import EfdtTree

import time
import datetime
import argparse
import pickle
from matplotlib import pyplot as plt


def arg_parse():
    parser = argparse.ArgumentParser(description='Incremental Decision Tree')
    parser.add_argument('--tree', nargs='+', type=str,
                        choices=['v', 'e', 'vfdt', 'efdt'], default=['v', 'e'])
    parser.add_argument('--dataset', type=str, default='forest')
    parser.add_argument('--max_instance', type=int, default=20000)
    parser.add_argument('--exp', type=str, default=datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d_%H:%M:%S'))
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()

    args = arg_parse()
    args.exp = args.dataset + '_' + args.exp

    if args.dataset == 'poker':
        attrTypes = [AttrType.CATE] * 10
    else:
        attrTypes = None
    stream = DataStream(dataset.config.csv_path[args.dataset], attrTypes=attrTypes, shuffle=True)
    candidate_attr, n_class = stream.attributes, stream.n_class

    for attr in candidate_attr:
        attr.print()

    # hyperparameter
    delta = 0.01
    max_depth = 100
    min_sample = 5
    tau = 0.05
    def metric_func(class_freq): return -gini(class_freq)

    models = []
    legend = []
    if 'v' in args.tree or 'vfdt' in args.tree:
        models.append(VfdtTree)
        legend.append('VFDT')
    if 'e' in args.tree or 'efdt' in args.tree:
        models.append(EfdtTree)
        legend.append('EFDT')

    learners = []
    for i, model in enumerate(models):
        learners.append(model(candidate_attr, n_class,
                              delta, max_depth, min_sample, tau))

    def output(performances):
        output_path = 'outputs/%s.pickle' % args.exp
        with open(output_path, 'wb') as f:
            pickle.dump({
                'dataset': args.dataset,
                'shuffle': stream.shuffle,
                'seed': stream.seed,
                'delta': delta,
                'max_depth': max_depth,
                'min_sample': min_sample,
                'tau': tau,
                'learners': legend,
                'performances': performances,
            }, f)

    eval = EvaluatePrequential(
        stream, learners, metric_func, max_inst=args.max_instance, output_func=output)
    performances = eval.doMainTask()
    for line in performances:
        plt.plot(line)
    
    # for learner in learners:
    #     learner.print()

    plt.title("%s dataset" % args.dataset)
    plt.xlabel("instances \\times 1000")
    plt.ylabel("error rate")
    plt.legend(labels=legend)
    plt.show()

    print('Total time: ', time.time() - start_time)
