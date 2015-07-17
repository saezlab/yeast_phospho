import numpy as np
from scipy.stats.stats import spearmanr, pearsonr


def pearson(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = pearsonr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue


def spearman(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = spearmanr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue