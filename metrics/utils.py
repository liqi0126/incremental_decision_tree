import numpy as np

from model.utils import AttrType


def splitting_metric(attr_type, njk, metric_func, total_sample=None, class_freq=None):
    """
    :param attr_type: the type of splitting attribute.
    :param njk: conditional distribution of samples over splitting_attr. n[j] indicates the distribution of samples given splitting_attr=j
    :param metric_func: the evaluation function
    :param total_sample: the total number of samples, can also be computed from a double nested for loops of njk
    :param class_freq: distribution of samples, can also be computed from a for loop of njk
    :return: [best_metric, best_split]
             best_metric: m(D, S=s) = |D1|/|D|*m(D1) + |D2|/|D|*m(D2)
             best_split is the splitting point of metric.
    """
    if attr_type == AttrType.NONE:
        return [float('-inf'), None]

    if class_freq is None:
        class_freq = {}
        for j in njk:
            nk = njk[j]
            for k in nk:
                if k not in class_freq:
                    class_freq[k] = nk[k]
                else:
                    class_freq[k] += nk[k]

    if total_sample is None:
        total_sample = sum(class_freq.values())

    if attr_type == AttrType.NUME:
        return splitting_metric_nume(njk, metric_func, total_sample, class_freq)
    elif attr_type == AttrType.CATE:
        return splitting_metric_cate(njk, metric_func, total_sample, class_freq)


def splitting_metric_nume(njk, metric_func, total_sample, class_freq):
    best_metric = float('-inf')
    best_split = None

    sorted_value = np.array(sorted(njk.keys()))
    split_value = (sorted_value[:-1] + sorted_value[1:]) / 2

    D = total_sample
    D1 = 0
    D1_class_freq = {k: 0 for k in class_freq}
    for index in range(len(split_value)):
        nk = njk[sorted_value[index]]
        for k in nk:
            D1_class_freq[k] += nk[k]
            D1 += nk[k]

        D2 = D - D1
        D2_class_freq = {k: class_freq[k] - D1_class_freq[k] for k in class_freq}

        m = 0
        if D1 > 0:
            m += D1/D * metric_func(D1_class_freq)
        if D2 > 0:
            m += D2/D * metric_func(D2_class_freq)

        if best_metric < m:
            best_metric = m
            best_split = split_value[index]

    return [best_metric, best_split]


def splitting_metric_cate(njk, metric_func, total_sample, class_freq):
    best_metric = float('-inf')
    best_split = None

    D = total_sample
    for j in njk:
        D1 = 0
        D1_class_freq = {k: 0 for k in class_freq}

        for k in njk[j]:
            D1_class_freq[k] = njk[j][k]
            D1 += njk[j][k]

        D2 = D - D1
        D2_class_freq = {k: class_freq[k] - D1_class_freq[k] for k in class_freq}

        m = 0
        if D1 > 0:
            m += D1/D * metric_func(D1_class_freq)
        if D2 > 0:
            m += D2/D * metric_func(D2_class_freq)

        if best_metric < m:
            best_metric = m
            best_split = j

    return [best_metric, best_split]
