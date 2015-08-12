import numpy as np
from scipy.stats.stats import spearmanr, pearsonr
from pandas import DataFrame


def pearson(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = pearsonr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue, mask.sum()


def spearman(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = spearmanr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue, mask.sum()


def cohend(x, y):
    return (np.mean(x) - np.mean(y)) / (np.sqrt((np.std(x) ** 2 + np.std(y) ** 2) / 2))


def metric(func, x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    res = func(x[mask], y[mask]) if np.sum(mask) > 5 else np.NaN
    return res if np.isfinite(res) else [[res]]


def shuffle(df):
    col, idx = df.columns, df.index
    val = df.values
    shape = val.shape
    val_flat = val.flatten()
    np.random.shuffle(val_flat)
    return DataFrame(val_flat.reshape(shape), columns=col, index=idx)