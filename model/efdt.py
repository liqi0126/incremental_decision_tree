from .tree import ClsNode, ClsTree
from .vfdt import VfdtTree, VfdtNode
from .utils import AttrType, Attr, hoeffing_bound
from metrics.utils import splitting_metric

from copy import deepcopy


class EfdtNode(VfdtNode):
    def __init__(self, candidate_attr, parent, init_class_freq=None):
        super().__init__(candidate_attr, parent, init_class_freq)

    def current_split_metric(self, metric_func):
        attr = self.split_attr
        i = self.candidate_attr.index(attr)
        D = self.total_sample

        m = 0

        if attr.type == AttrType.NUME:
            D1 = 0
            D1_class_freq = {k: 0 for k in self.class_freq}

            for j in self.nijk[i]:
                if j <= self.split_value:
                    nk = self.nijk[i][j]
                    for k in nk:
                        D1_class_freq[k] += nk[k]
                        D1 += nk[k]

            D2 = D - D1
            D2_class_freq = {k: self.class_freq[k] -
                             D1_class_freq[k] for k in self.class_freq}

            if D1 > 0:
                m += D1/D * metric_func(D1_class_freq)
            if D2 > 0:
                m += D2/D * metric_func(D2_class_freq)

        elif attr.type == AttrType.CATE:
            njk = self.nijk[i]
            for j in njk:
                D1 = sum(njk[j].values())
                if D1 > 0:
                    m += D1/D * metric_func(njk[j])

        return m

    def trace_down_to_leaf(self, x):
        node = self
        path = [node]
        while not node.is_leaf():
            node = node.trace_down(x)
            path.append(node)
        return path

    def attempt_to_split(self, metric_func, n_class, delta, max_depth, min_sample, grace_period, tau=None):
        if len(self.candidate_attr) == 0:
            return
        if self.depth is not None and self.depth > max_depth:
            return
        if self.total_sample < min_sample:
            return
        if len(self.class_freq) == 1:
            return

        self.instance_count += 1
        if self.instance_count % grace_period != 0:
            return
        else:
            self.instance_count = 0

        metric0 = metric_func(self.class_freq)

        best_split_attr = None
        best_split_value = None
        best_metric_val = float('-inf')
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

        epsilon = hoeffing_bound(metric_func, n_class,
                                 delta, self.total_sample)

        # if best_metric_val - metric0 > epsilon:
        # if best_metric_val - metric0 > epsilon or (tau is not None and 0 < best_metric_val - metric0 < epsilon < tau):
        if best_metric_val - metric0 > epsilon or (tau is not None and epsilon < tau):
            self.split(best_split_attr, best_split_value)

    def reevaluate_best_split(self, metric_func, n_class, delta, min_samples_reevaluate, tau=None):
        if len(self.candidate_attr) == 0:
            return

        self.instance_count += 1
        if self.instance_count % min_samples_reevaluate != 0:
            return
        else:
            self.instance_count = 0

        metric0 = metric_func(self.class_freq)
        current_metric = self.current_split_metric(metric_func)

        best_split_attr = None
        best_split_value = None
        best_metric_val = float('-inf')
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

        epsilon = hoeffing_bound(metric_func, n_class,
                                 delta, self.total_sample)

        if metric0 > best_metric_val:
            if metric0 - current_metric > epsilon or (tau is not None and epsilon < tau and metric0 - current_metric > tau/2):
                self.cut()
                return True
        else:
            if best_metric_val - current_metric > epsilon or (tau is not None and epsilon < tau and best_metric_val - current_metric > tau/2):
                self.split(best_split_attr, best_split_value)
                return True

        return False

    def cut(self):
        self.children = []

    def split(self, best_split_attr, best_split_value):
        # TODO: binary split, should we discard splitting attribute?
        # TODO: how to do prediction for leaf after just splitting?
        self.split_attr = best_split_attr
        self.split_value = best_split_value

        if best_split_attr.type == AttrType.CATE:
            # candidate_attr = deepcopy(self.candidate_attr)
            # candidate_attr.pop(self.candidate_attr.index(best_split_attr))
            # self.children = [EfdtNode(candidate_attr, self) for v in best_split_attr.values]

            self.children = []
            for _ in best_split_attr.values:
                candidate_attr = deepcopy(self.candidate_attr)
                candidate_attr.pop(self.candidate_attr.index(best_split_attr))
                self.children.append(EfdtNode(
                    candidate_attr, self))
        elif best_split_attr.type == AttrType.NUME:
            self.children = [EfdtNode(deepcopy(self.candidate_attr), self), EfdtNode(
                deepcopy(self.candidate_attr), self)]
        else:
            raise NotImplementedError


class EfdtTree(VfdtTree):
    def __init__(self, candidate_attr, n_class, delta, min_samples_reevaluate=20, grace_period=100, max_depth=100, min_sample=5, tau=None):
        super().__init__(candidate_attr, n_class, delta, grace_period, max_depth, min_sample, tau)
        self.root = EfdtNode(candidate_attr, parent=None)
        self.min_samples_reevaluate = min_samples_reevaluate

    def _update(self, _x, _y, metric_func):
        path = self.root.trace_down_to_leaf(_x)
        for node in path:
            node.add_sample(_x, _y)
            if node.is_leaf():
                node.attempt_to_split(
                    metric_func, self.n_class, self.delta, self.max_depth, self.min_sample, self.grace_period, self.tau)
            else:
                success = node.reevaluate_best_split(metric_func, self.n_class, self.delta, self.min_samples_reevaluate, self.tau)
                if success:
                    break

    def _predict(self, x):
        return self.root.trace_down_to_leaf(x)[-1].most_freq()
