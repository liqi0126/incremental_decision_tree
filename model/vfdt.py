from .tree import ClsNode, ClsTree
from .utils import AttrType, Attr, hoeffing_bound
from metrics.utils import splitting_metric

from copy import deepcopy


class VfdtNode(ClsNode):
    def __init__(self, candidate_attr, parent, init_class_freq=None):
        super().__init__(candidate_attr, parent)
        # TODO: how to deal with continuous value?
        self.nijk = [{} for attr in candidate_attr]
        if init_class_freq is not None:
            self.class_freq = init_class_freq

    def add_sample(self, x, y):
        if y not in self.class_freq:
            self.class_freq[y] = 1
        else:
            self.class_freq[y] += 1

        self.total_sample += 1

        for i, attr in enumerate(self.candidate_attr):
            j = x[attr.index]
            if j not in self.nijk[i]:
                self.nijk[i][j] = {y: 1}
            else:
                if y not in self.nijk[i][j]:
                    self.nijk[i][j][y] = 1
                else:
                    self.nijk[i][j][y] += 1

    def attempt_to_split(self, metric_func, n_class, delta, max_depth, min_sample, tau=None):
        if len(self.candidate_attr) == 0: return

        if self.depth > max_depth:
            return

        if self.total_sample < min_sample:
            return

        if len(self.class_freq) == 1:
            return

        # TODO: what G_m means in the paper?
        metric0 = metric_func(self.class_freq)

        best_split_attr = None
        best_split_value = None
        best_metric_val = float('-inf')
        second_metric_val = float('-inf')
        for i, attr in enumerate(self.candidate_attr):
            if attr.type == AttrType.NONE:
                continue
            njk = self.nijk[i]
            split_metric, split_value = splitting_metric(
                attr.type, njk, metric_func, self.total_sample, self.class_freq)
            if split_metric > best_metric_val:
                best_metric_val = split_metric
                best_split_attr = attr
                best_split_value = split_value
            elif best_metric_val > split_metric > second_metric_val:
                second_metric_val = split_metric

        epsilon = hoeffing_bound(metric_func, n_class,
                                 delta, self.total_sample)

        if best_metric_val > metric0:
            if best_metric_val > second_metric_val + epsilon:
                self.split(best_split_attr, best_split_value)
            # Ties
            elif tau is not None and best_metric_val - second_metric_val < epsilon < tau:
                self.split(best_split_attr, best_split_value)

    def split(self, best_split_attr, best_split_value):
        # TODO: binary split, should we discard splitting attribute?
        # TODO: how to do prediction for leaf after just splitting?
        self.split_attr = best_split_attr
        self.split_value = best_split_value

        if best_split_attr.type == AttrType.CATE:
            candidate_attr = deepcopy(self.candidate_attr)
            candidate_attr.pop(self.candidate_attr.index(best_split_attr))
            self.children = [VfdtNode(candidate_attr, self)
                             for v in best_split_attr.values]
        elif best_split_attr.type == AttrType.NUME:
            self.children = [VfdtNode(deepcopy(self.candidate_attr), self), VfdtNode(
                deepcopy(self.candidate_attr), self)]
        else:
            raise NotImplementedError


class VfdtTree(ClsTree):
    def __init__(self, candidate_attr, n_class, delta, max_depth=100, min_sample=5, tau=None):
        super().__init__(max_depth=max_depth, min_sample=min_sample)
        self.root = VfdtNode(candidate_attr, parent=None)

        self.n_class = n_class
        self.delta = delta
        self.tau = tau

    def update(self, X, y, metric_func):
        for (_x, _y) in zip(X, y):
            self._update(_x, _y, metric_func)

    def _update(self, _x, _y, metric_func):
        leaf = self.root.trace_down_to_leaf(_x)
        leaf.add_sample(_x, _y)
        leaf.attempt_to_split(metric_func, self.n_class,
                              self.delta, self.max_depth, self.min_sample, self.tau)
