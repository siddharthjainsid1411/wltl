# wtltl.py
import numpy as np

def weighted_and(w, x):
    w = np.array(w)
    x = np.array(x)
    w_bar = w / np.sum(w)
    vals = []
    for i in range(len(x)):
        term = ((0.5 - w_bar[i]) * np.sign(x[i]) + 0.5) * x[i]
        vals.append(term)
    return np.min(vals)

def weighted_or(w, x):
    return -weighted_and(w, -np.array(x))