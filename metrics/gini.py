
def gini(class_freq, total=None):
    g = 1
    if total is None:
        total = sum(class_freq.values())
    for k in class_freq.values():
        g -= (k/total)**2
    return g
