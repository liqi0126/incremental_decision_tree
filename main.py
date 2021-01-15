from dataset.stream import DataStream
import dataset.config
from evaluation.EvaluatePrequential import EvaluatePrequential

from metrics.metrics import gini, infogain

from model.utils import AttrType
from model.vfdt import VfdtTree
from model.efdt import EfdtTree

from utils import *

import time
import datetime
import yaml
import argparse
import pickle
from matplotlib import pyplot as plt
import numpy as np
import random
from copy import deepcopy


def arg_parse():
    parser = argparse.ArgumentParser(description='Incremental Decision Tree')

    # Trees
    parser.add_argument('--tree', required=True, nargs='+', type=str,
                        choices=['v', 'e', 'vfdt', 'efdt'], default=['v', 'e'],
                        help="Tree models to use")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to yaml config file")

    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset to use")
    parser.add_argument("--seed", type=int, default=4096,
                        help="Random seed")
    parser.add_argument("--shuffle", action='store_true',
                        help="Whether to shuffle data stream")

    # Experiments
    parser.add_argument('--exp', type=str, 
                        default=datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H:%M:%S'),
                        help="Experiments identifier, used to name output file")
    parser.add_argument("--plot", action='store_true',
                        help="Whether to plot error rates after mining")
    parser.add_argument("--verbose", action='store_true',
                        help="Whether to output progress information")

    # Post process
    args = parser.parse_args()

    args.exp = args.dataset + '_' + args.exp
    if args.shuffle:
        args.exp += '_shuffle'
        args.exp += f'_seed{args.seed}'
    else:
        args.exp += '_unshuffle'

    if 'v' in args.tree or 'vfdt' in args.tree:
        args.exp = 'vfdt_' + args.exp
    if 'e' in args.tree or 'efdt' in args.tree:
        args.exp = 'efdt_' + args.exp

    if args.config is None:
        args.config = dataset.config.datasets_config[args.dataset]['yml_config']
    with open(args.config) as f:
        args.config = yaml.load(f.read(), Loader=yaml.FullLoader)

    return args


if __name__ == '__main__':
    start_time = time.time()

    args = arg_parse()

    np.random.seed(args.seed)
    random.seed(args.seed)

    """
    Dataset
    """
    if 'attr_types' in dataset.config.datasets_config[args.dataset]:
        attrTypes = dataset.config.datasets_config[args.dataset]['attr_types']
    else:
        attrTypes = None
    stream = DataStream(dataset.config.datasets_config[args.dataset]['csv_path'],
                        attrTypes=attrTypes, shuffle=args.shuffle, seed=args.seed)
    candidate_attr, n_class = stream.attributes, stream.n_class

    if args.verbose:
        print("All attributes: ")
        for attr in candidate_attr:
            attr.print()

    """
    Trees
    """
    legend = []
    learners = []

    if 'v' in args.tree or 'vfdt' in args.tree:
        learners.append(VfdtTree(
            candidate_attr=deepcopy(candidate_attr),
            n_class=n_class,
            delta=args.config['delta'],
            nume_max_class=args.config['nume_max_class'],
            grace_period=args.config['grace_period'],
            max_depth=args.config['max_depth'],
            tau=args.config['tau']
        ))
        legend.append('VFDT')

    if 'e' in args.tree or 'efdt' in args.tree:
        learners.append(EfdtTree(
            candidate_attr=deepcopy(candidate_attr),
            n_class=n_class,
            delta=args.config['delta'],
            nume_max_class=args.config['nume_max_class'],
            min_samples_reevaluate=args.config['min_samples_reevaluate'],
            grace_period=args.config['grace_period'],
            max_depth=args.config['max_depth'],
            tau=args.config['tau']
        ))
        legend.append('EFDT')

    """
    Train
    """
    if args.config['metric'] == 'gini':
        def metric_func(class_freq):
            return - gini(np.fromiter(class_freq.values(), dtype=int))
    elif args.config['metric'] == 'infogain':
        def metric_func(class_freq):
            return - infogain(np.fromiter(class_freq.values(), dtype=int))
    else:
        raise NotImplementedError

    def output(performances):
        output_path = 'outputs/%s.pickle' % args.exp
        ensure_parent_dir(output_path)
        with open(output_path, 'wb') as f:
            pickle.dump({
                'dataset': args.dataset,
                'shuffle': stream.shuffle,
                'seed': stream.seed,
                'args': args,
                'learners': legend,
                'performances': performances,
                'total_time': time.time() - start_time
            }, f)

    eval = EvaluatePrequential(
        stream,
        learners,
        metric_func,
        max_inst=args.config['max_instance'],
        output_func=output
    )
    performances = eval.doMainTask()

    print('Total time: ', time.time() - start_time)

    if args.plot:
        for line in performances:
            plt.plot(line)
        plt.title("%s dataset" % args.dataset)
        plt.xlabel("Instances (x 1,000)")
        plt.ylabel("Error rate")
        plt.legend(labels=legend)
        plt.show()
