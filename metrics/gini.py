from numba import jit
import numpy as np

@jit(nopython=True)
def gini(class_freq, total=None):
    if total is None:
        total = np.sum(class_freq)
    return 1. - np.sum((class_freq/total)**2)
