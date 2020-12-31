from dataset.stream import DataStream
import dataset.config
from evaluation.EvaluatePrequential import EvaluatePrequential

from metrics.gini import gini

from model.utils import Attr, AttrType
from model.tree import ClsTree
from model.vfdt import VfdtTree
from model.efdt import EfdtTree

import time
import argparse
from matplotlib import pyplot as plt 


def arg_parse():
    parser = argparse.ArgumentParser(description='Incremental Decision Tree')
    parser.add_argument('--tree', nargs='+', type=str,
                        choices=['v', 'e', 'vfdt', 'efdt'], default=['v', 'e'])
    parser.add_argument('--dataset', type=str, default='forest')
    parser.add_argument('--max_instance', type=int, default=20000)
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()

    args = arg_parse()
    
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
    max_depth = 20
    min_sample = 5
    tau = 0.05
    def metric_func(class_freq): return -gini(class_freq)

    learners = []
    legend = []
    if 'v' in args.tree or 'vfdt' in args.tree:
        learners.append(VfdtTree)
        legend.append('VFDT')
    if 'e' in args.tree or 'efdt' in args.tree:
        learners.append(EfdtTree)
        legend.append('EFDT')
    
    for i, learner in enumerate(learners):
        print(legend[i])
        model = learner(candidate_attr, n_class, delta, max_depth, min_sample, tau)
        eval = EvaluatePrequential(stream, model, metric_func, max_inst=args.max_instance)
        performance = eval.doMainTask()
        plt.plot(performance)
        # model.print()

    plt.title("%s dataset" % args.dataset) 
    plt.xlabel("instances \\times 1000") 
    plt.ylabel("error rate") 
    plt.legend(labels=legend)
    plt.show()

    print('Total time: ', time.time() - start_time)
