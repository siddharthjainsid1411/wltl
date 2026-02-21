# smooth.py
import numpy as np

def smooth_min(values, k=10.0):
    values = np.array(values)
    return -1.0/k * np.log(np.sum(np.exp(-k*values)))

def smooth_max(values, k=10.0):
    values = np.array(values)
    weights = np.exp(k*values)
    return np.sum(values * weights) / np.sum(weights)