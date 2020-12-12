from .tree import ClsNode, ClsTree
from .vfdt import VfdtTree, VfdtNode
from .utils import AttrType, Attr, hoeffing_bound
from metrics.utils import splitting_metric


class EfdtNode(VfdtNode):
    def __init__(self, candidate_attr, parent):
        super().__init__(candidate_attr, parent)

    def current_split_metric(self, metric_func):
        attr = self.split_attr
        D = self.total_sample

        D1 = 0
        D1_class_freq = {k: 0 for k in self.class_freq}

        if attr.type == AttrType.NUME:
            for j in self.nijk[attr.name]:
                if j <= self.split_value:
                    nk = self.nijk[attr.name][j]
                    for k in nk:
                        D1_class_freq[k] += nk[k]
                        D1 += nk[k]
        elif attr.type == AttrType.CATE:
            njk = self.nijk[attr.name]
            j = self.split_value
            for k in njk[j]:
                D1_class_freq[k] = njk[j][k]
                D1 += njk[j][k]

        D2 = D - D1
        D2_class_freq = {k: self.class_freq[k] -
                         D1_class_freq[k] for k in self.class_freq}

        m = 0
        if D1 > 0:
            m += D1/D * metric_func(D1_class_freq)
        if D2 > 0:
            m += D2/D * metric_func(D2_class_freq)
        return m

    def trace_down_to_leaf(self, x):
        node = self
        path = [node]
        while not node.is_leaf():
            node = node.trace_down(x)
            path.append(node)
        return path

    def attempt_to_split(self, metric_func, n_class, delta, max_depth, min_sample):
        if self.depth > max_depth:
            return
        if self.total_sample < min_sample:
            return
        if len(self.class_freq) == 1:
            return

        metric0 = metric_func(self.class_freq)

        best_split_attr = None
        best_split_value = None
        best_metric_val = float('-inf')
        for attr in self.candidate_attr:
            if attr.type == AttrType.NONE:
                continue
            njk = self.nijk[attr.name]
            split_metric, split_value = splitting_metric(
                attr.type, njk, metric_func, self.total_sample, self.class_freq)
            if split_metric > best_metric_val:
                best_metric_val = split_metric
                best_split_attr = attr
                best_split_value = split_value

        epsilon = hoeffing_bound(metric_func, n_class,
                                 delta, self.total_sample)

        if best_metric_val - metric0 > epsilon:
            self.split(best_split_attr, best_split_value)

    def reevaluate_best_split(self, metric_func, n_class, delta):
        current_metric = self.current_split_metric(metric_func)

        best_split_attr = None
        best_split_value = None
        best_metric_val = float('-inf')
        for attr in self.candidate_attr:
            if attr.type == AttrType.NONE:
                continue
            njk = self.nijk[attr.name]
            split_metric, split_value = splitting_metric(
                attr.type, njk, metric_func, self.total_sample, self.class_freq)
            if split_metric > best_metric_val:
                best_metric_val = split_metric
                best_split_attr = attr
                best_split_value = split_value

        epsilon = hoeffing_bound(metric_func, n_class,
                                 delta, self.total_sample)

        if best_metric_val - current_metric > epsilon:
            self.split(best_split_attr, best_split_value)

    def split(self, best_split_attr, best_split_value):
        # TODO: binary split, should we discard splitting attribute?
        # TODO: how to do prediction for leaf after just splitting?
        self.split_attr = best_split_attr
        self.split_value = best_split_value
        self.left_child = EfdtNode(self.candidate_attr, self)
        self.right_child = EfdtNode(self.candidate_attr, self)


class EfdtTree(VfdtTree):
    def __init__(self, candidate_attr, n_class, delta, max_depth=100, min_sample=5):
        super().__init__(candidate_attr, n_class, delta, max_depth, min_sample)
        self.root = EfdtNode(candidate_attr, parent=None)

    def _update(self, _x, _y, metric_func):
        path = self.root.trace_down_to_leaf(_x)
        for node in path:
            node.add_sample(_x, _y)
            if node.is_leaf():
                node.attempt_to_split(
                    metric_func, self.n_class, self.delta, self.max_depth, self.min_sample)
            else:
                success = node.reevaluate_best_split(
                    metric_func, self.n_class, self.delta)
                if success:
                    break

    def _predict(self, x):
        return self.root.trace_down_to_leaf(x)[-1].most_freq()
