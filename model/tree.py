from collections import Counter
import numpy as np

from model.utils import AttrType, Attr
from metrics.utils import splitting_metric


class ClsNode:
    def __init__(self, candidate_attr, parent):
        self.parent = parent
        self.left_child = None
        self.right_child = None

        self.candidate_attr = candidate_attr

        self.depth = parent.depth if parent is not None else 0

        self.split_attr = None
        self.split_value = None

        self.total_sample = 0
        self.class_freq = {}

        self.prediction = None

    # given example x, trace down to child
    def trace_down(self, x):
        if self.is_leaf():
            return self

        value = x[self.split_attr.index]
        if self.split_attr.type == AttrType.CATE:
            if value == self.split_value:
                return self.left_child
            else:
                return self.right_child
        elif self.split_attr.type == AttrType.NUME:
            if value <= self.split_value:
                return self.left_child
            else:
                return self.right_child
        else:
            raise RuntimeError

    # trace all the way down to leaf
    def trace_down_to_leaf(self, x):
        node = self
        while not node.is_leaf():
            node = node.trace_down(x)
        return node

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    def most_freq(self):
        try:
            return max(self.class_freq, key=self.class_freq.get)
        except ValueError:
            return self.parent.most_freq()

    def recur_splitting(self, X, y, metric_func, max_depth, min_sample):
        self.total_sample = len(y)
        self.class_freq = Counter(y)

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

        # TODO: optimization
        for attr in self.candidate_attr:
            if attr.type == AttrType.NONE:
                continue

            njk = {}
            for (_x, k) in zip(X, y):
                j = _x[attr.index]
                if j not in njk:
                    njk[j] = {k: 1}
                else:
                    if k not in njk[j]:
                        njk[j][k] = 1
                    else:
                        njk[j][k] += 1

            split_metric, split_value = splitting_metric(attr.type, njk, metric_func, self.total_sample, self.class_freq)
            # TODO: we can also use hoeffding bound to split ?
            if split_metric > best_metric_val:
                best_metric_val = split_metric
                best_split_attr = attr
                best_split_value = split_value

        if best_metric_val < metric0:
            return

        if best_split_attr.type == AttrType.CATE:
            left_index = np.array(X[:, best_split_attr.index]) == best_split_value
        elif best_split_attr.type == AttrType.NUME:
            left_index = np.array(X[:, best_split_attr.index]) < best_split_value
        else:
            raise NotImplementedError

        right_index = ~left_index
        X_left, X_right, y_left, y_right = X[left_index], X[right_index], y[left_index], y[right_index]

        self.split_attr = best_split_attr
        self.split_value = best_split_value
        self.right_child = ClsNode(self.candidate_attr, self)
        self.left_child = ClsNode(self.candidate_attr, self)
        self.right_child.recur_splitting(X_right, y_right, metric_func, max_depth, min_sample)
        self.left_child.recur_splitting(X_left, y_left, metric_func, max_depth, min_sample)

    def print(self):
        head = '    ' * self.depth
        if self.is_leaf():
            print(head + str(self.most_freq()))
        else:
            if self.split_attr.type == AttrType.CATE:
                left_symbol = '=='
                right_symbol = '!='
            else:
                left_symbol = '<'
                right_symbol = '>='

            print(head + str(self.split_attr.name) + left_symbol + str(self.split_value))
            self.left_child.print()
            print(head + str(self.split_attr.name) + right_symbol + str(self.split_value))
            self.right_child.print()


class ClsTree:
    def __init__(self, candidate_attr=None, max_depth=100, min_sample=5):
        self.root = ClsNode(candidate_attr, parent=None)
        self.max_depth = max_depth
        self.min_sample = min_sample

    def fit(self, X, y, metric_func):
        self.root.recur_splitting(X, y, metric_func, self.max_depth, self.min_sample)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        return self.root.trace_down_to_leaf(x).most_freq()

    def print(self):
        self.root.print()
