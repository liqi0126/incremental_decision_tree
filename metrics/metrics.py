from numba import jit
import numpy as np

@jit(nopython=True)
def gini(class_freq, total=None):
    if total is None:
        total = np.sum(class_freq)
    return 1. - np.sum((class_freq/total)**2)


@jit(nopython=True)
def infogain(class_freq, total=None):
    if total is None:
        total = np.sum(class_freq)
    class_freq = class_freq / total
    class_freq = class_freq[class_freq > 0]
    return - np.sum(class_freq * np.log(class_freq) / np.log(2))
