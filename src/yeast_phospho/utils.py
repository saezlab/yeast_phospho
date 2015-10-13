import numpy as np
from yeast_phospho import wd
from scipy.stats.stats import spearmanr, pearsonr
from pandas import DataFrame, read_csv, pivot_table


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


def count_percentage(df):
    return float(df.count().sum()) / np.prod(df.shape) * 100


def get_kinases_targets(studies_to_filter={'21177495'}):
    k_targets = read_csv('%s/files/PhosphoGrid.txt' % wd, sep='\t')

    k_targets = k_targets[[len(set(i.split('|')).intersection(studies_to_filter)) != len(set(i.split('|'))) for i in k_targets['KINASES_EVIDENCE_PUBMED']]]
    k_targets = k_targets[[len(set(i.split('|')).intersection(studies_to_filter)) != len(set(i.split('|'))) for i in k_targets['PHOSPHATASES_EVIDENCE_PUBMED']]]

    k_targets = k_targets.loc[(k_targets['KINASES_ORFS'] != '-') | (k_targets['PHOSPHATASES_ORFS'] != '-')]

    k_targets['SOURCE'] = k_targets['KINASES_ORFS'] + '|' + k_targets['PHOSPHATASES_ORFS']
    k_targets = [(k, t + '_' + site) for t, site, source in k_targets[['ORF_NAME', 'PHOSPHO_SITE', 'SOURCE']].values for k in source.split('|') if k != '-' and k != '']
    k_targets = DataFrame(k_targets, columns=['kinase', 'site'])

    k_targets['value'] = 1

    k_targets = pivot_table(k_targets, values='value', index='site', columns='kinase', fill_value=0)

    print '[INFO] [PHOSPHOGRID] Kinases targets: ', k_targets.shape

    return k_targets
