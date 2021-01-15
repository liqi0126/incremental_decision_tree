from .tree import ClsNode, ClsTree
from .utils import AttrType, Attr, hoeffing_bound
from metrics.utils import splitting_metric

from copy import deepcopy


class VfdtNode(ClsNode):
    def __init__(self, candidate_attr, parent, init_class_freq=None):
        super().__init__(candidate_attr, parent)

        # node statistic
        self.nijk = [{} for _ in candidate_attr]
        self.nume_list = {attr.name: {'value': [], 'label': []} for attr in candidate_attr if attr.type == AttrType.NUME}

        self.build_nume_threshold = 1
        self.nume_count = 0

        if init_class_freq is not None:
            self.class_freq = deepcopy(init_class_freq)
        self.instance_count = 0


    def trace_down(self, x):
        if self.is_leaf():
            return self

        value = x[self.split_attr.index]
        if self.split_attr.type == AttrType.CATE:
            return self.children[self.split_attr.values.index(value)]
        elif self.split_attr.type == AttrType.NUME:
            return self.children[self.get_nume_key(value, self.split_attr.values)[0]]
        else:
            raise RuntimeError


    def add_sample(self, x, y, nume_max_class):
        if y not in self.class_freq:
            self.class_freq[y] = 1
        else:
            self.class_freq[y] += 1

        self.total_sample += 1
        self.nume_count += 1

        for i, attr in enumerate(self.candidate_attr):
            j = x[attr.index]
            if attr.type == AttrType.CATE:
                if j not in self.nijk[i]:
                    self.nijk[i][j] = {y: 1}
                else:
                    if y not in self.nijk[i][j]:
                        self.nijk[i][j][y] = 1
                    else:
                        self.nijk[i][j][y] += 1
            else:
                attr_nume = self.nume_list[attr.name]
                attr_nume['value'].append(j)
                attr_nume['label'].append(y)
                if self.nume_count == self.build_nume_threshold:
                    self.nijk[i] = self.build_nume_dict(attr_nume, nume_max_class)
        
        if self.nume_count == self.build_nume_threshold:
            self.build_nume_threshold *= 2
                

    def attempt_to_split(self, metric_func, n_class, delta, max_depth, grace_period, nume_max_class, tau=None):
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
        second_metric_val = float('-inf')

        for i, attr in enumerate(self.candidate_attr):
            if attr.type == AttrType.NONE:
                continue

            split_metric, split_value = splitting_metric(
                attr.type, self.nijk[i], metric_func, self.total_sample, self.class_freq)
            if split_metric > best_metric_val:
                second_metric_val = best_metric_val
                best_metric_val = split_metric
                best_split_attr = attr
                best_split_value = split_value
            elif best_metric_val > split_metric > second_metric_val:
                second_metric_val = split_metric

        epsilon = hoeffing_bound(metric_func, n_class,
                                 delta, self.total_sample)

        if best_metric_val > null_metric:
            if (best_metric_val > second_metric_val + epsilon) or (tau is not None and epsilon < tau):
                self.split(best_split_attr, best_split_value, NodeType=VfdtNode)


    def split(self, best_split_attr, best_split_value, NodeType):
        self.split_attr = best_split_attr
        self.split_value = best_split_value

        njk = self.nijk[self.candidate_attr.index(best_split_attr)]

        if best_split_attr.type == AttrType.CATE:
            candidate_attr = deepcopy(self.candidate_attr)
            candidate_attr.pop(self.candidate_attr.index(best_split_attr))
            self.children = []
            for v in best_split_attr.values:
                self.children.append(NodeType(
                        deepcopy(candidate_attr), 
                        parent=self,
                    )
                )
        elif best_split_attr.type == AttrType.NUME:
            best_split_attr.values = list(njk.keys())
            self.children = []
            for v in best_split_attr.values:
                self.children.append(NodeType(
                        deepcopy(self.candidate_attr), 
                        parent=self,
                    )
                )
        else:
            raise NotImplementedError


class VfdtTree(ClsTree):
    """VFDT: Hoeffding Tree Implementation

    Parameters
    ----------
    candidate_attr:
        List of Attrs to generate split suggestions
    n_class:
        The number of different classes
    delta:
        Confidence in Hoeffding bound
    nume_max_class:
        The maximum number of bins in histogram observers for NUME attributes.
    grace_period:
        Number of instances observed between split attempts.
    max_depth:
        Maximum depth of a tree
    tau:
        Tie threshold, see VFDT paper for more details
    """
    def __init__(
        self, 
        candidate_attr, 
        n_class, 
        delta=1e-7, 
        nume_max_class=10, 
        grace_period=100, 
        max_depth=100, 
        tau=0.05
    ):
        super().__init__(max_depth=max_depth)
        self.root = VfdtNode(candidate_attr, parent=None)
        self.grace_period = grace_period
        self.n_class = n_class
        self.delta = delta
        self.tau = tau
        self.nume_max_class = nume_max_class

    def update(self, X, y, metric_func):
        for (_x, _y) in zip(X, y):
            self._update(_x, _y, metric_func)

    def _update(self, _x, _y, metric_func):
        leaf = self.root.trace_down_to_leaf(_x)
        leaf.add_sample(_x, _y, self.nume_max_class)
        leaf.attempt_to_split(metric_func, self.n_class, self.delta, self.max_depth, self.grace_period, self.nume_max_class, self.tau)

    def learn_one(self, x, y, metric_func):
        self._update(x, y, metric_func)