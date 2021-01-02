from numba import jit
import numpy as np

@jit(nopython=True)
def gini(class_freq, total=None):
    g = np.float(1.)
    if total is None:
        total = np.sum(class_freq)
    for k in class_freq:
        g -= (k/total)**2
    return g
