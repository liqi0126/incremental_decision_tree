from .tree import ClsNode, ClsTree
from .vfdt import VfdtTree, VfdtNode
from .utils import AttrType, Attr, hoeffing_bound
from metrics.utils import splitting_metric

from copy import deepcopy


class EfdtNode(VfdtNode):
    def __init__(self, candidate_attr, parent, init_class_freq=None):
        super().__init__(candidate_attr, parent, init_class_freq)

    def trace_down_to_leaf(self, x):
        node = self
        path = [node]
        while not node.is_leaf():
            node = node.trace_down(x)
            path.append(node)
        return path

    def attempt_to_split(self, metric_func, n_class, delta, max_depth, grace_period, tau=None):
        if len(self.candidate_attr) == 0:
            return
        if self.depth is not None and self.depth > max_depth:
            return
        if len(self.class_freq) == 1:
            return      

        self.instance_count += 1
        if self.instance_count % grace_period != 0:
            return
        else:
            self.instance_count = 0  

        null_metric = metric_func(self.class_freq)

        best_split_attr = None
        best_split_value = None
        best_metric_val = float('-inf')

        for i, attr in enumerate(self.candidate_attr):
            if attr.type == AttrType.NONE:
                continue

            njk = self.nijk[i]
            split_metric, split_value = splitting_metric(
                attr.type, njk, metric_func, self.total_sample, class_freq=None)
            if split_metric > best_metric_val:
                best_metric_val = split_metric
                best_split_attr = attr
                best_split_value = split_value

        epsilon = hoeffing_bound(metric_func, n_class,
                                 delta, self.total_sample)

        if best_metric_val - null_metric > epsilon or (tau is not None and epsilon < tau):
            self.split(best_split_attr, best_split_value, NodeType=EfdtNode)
            print('split at leaf', self.depth)


    def reevaluate_best_split(self, metric_func, n_class, delta, min_samples_reevaluate, tau=None):
        if len(self.candidate_attr) == 0:
            return False

        self.instance_count += 1
        if self.instance_count % min_samples_reevaluate != 0:
            return False
        else:
            self.instance_count = 0

        null_metric = metric_func(self.class_freq)

        best_split_attr = None
        best_split_value = None
        best_metric_val = float('-inf')
        for i, attr in enumerate(self.candidate_attr):
            if attr.type == AttrType.NONE:
                continue

            njk = self.nijk[i]
            split_metric, split_value = splitting_metric(
                attr.type, njk, metric_func, self.total_sample, class_freq=None)
            if split_metric > best_metric_val:
                best_metric_val = split_metric
                best_split_attr = attr
                best_split_value = split_value
            
            if self.split_attr == attr:
                current_metric = split_metric

        epsilon = hoeffing_bound(metric_func, n_class,
                                 delta, self.total_sample)

        if null_metric - best_metric_val > epsilon:
            print('cut')
            self.cut()
            return True
        else:
            if (
                best_metric_val - current_metric > epsilon or (tau is not None and epsilon < tau)
            ) and (best_split_attr != self.split_attr):
                print('split', self.depth)
                self.split(best_split_attr, best_split_value, NodeType=EfdtNode)
                return True

        return False

    def cut(self):
        self.children = []


class EfdtTree(VfdtTree):
    def __init__(self, candidate_attr, n_class, delta=1e-7, min_samples_reevaluate=20, grace_period=100, max_depth=100, tau=0.05):
        super().__init__(candidate_attr, n_class=n_class, delta=delta, grace_period=grace_period, max_depth=max_depth, tau=tau)
        self.root = EfdtNode(candidate_attr, parent=None)
        self.min_samples_reevaluate = min_samples_reevaluate

    def _update(self, _x, _y, metric_func):
        path = self.root.trace_down_to_leaf(_x)
        for node in path:
            node.add_sample(_x, _y, self.nume_max_class)
            if node.is_leaf():
                node.attempt_to_split(
                    metric_func, self.n_class, self.delta, self.max_depth, self.grace_period, self.tau)
            else:
                success = node.reevaluate_best_split(metric_func, self.n_class, self.delta, self.min_samples_reevaluate, self.tau)
                if success:
                    print('success')
                    break
                pass

    def _predict(self, x):
        return self.root.trace_down_to_leaf(x)[-1].most_freq()
