from __future__ import division
import itertools as it
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from yeast_phospho import wd
from pandas import DataFrame, Series, read_csv
from scipy.stats.stats import pearsonr


def pearson(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = pearsonr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue, np.sum(mask)

# Import kinase activity
kinase_df = read_csv(wd + 'tables/kinase_enrichment_df.tab', sep='\t', index_col=0)


# Randomly suffle kinase matrix while keeping NaNs position
def randomise_matrix(matrix):
    random_df = matrix.copy()
    movers = ~np.isnan(random_df.values)
    random_df.values[movers] = np.random.permutation(random_df.values[movers])
    return random_df

n_permutations = 10000
random_kinases = [randomise_matrix(kinase_df) for i in xrange(n_permutations)]
print '[INFO] Kinase activity randomisation done: ', len(random_kinases)


def empirical_pvalue(p1, p2, verbose=1):
    # Calculate analytical correlaiton p-value
    cor, a_pvalue, n_meas = pearson(kinase_df.ix[p1], kinase_df.ix[p2])

    # Calculate correlaiton of randomised matrices
    random_cor = [pearson(r_matrix.ix[p1], r_matrix.ix[p2])[0] for r_matrix in random_kinases]
    count = sum([(r_cor >= cor >= 0) or (r_cor <= cor < 0) for r_cor in random_cor])

    # Calculate empirical p-value
    e_pvalue = 1 / n_permutations if count == 0 else count / n_permutations

    if verbose > 0:
        print '[INFO] ', p1, p2, cor, a_pvalue, e_pvalue, n_meas

    return p1, p2, cor, a_pvalue, e_pvalue, n_meas

p1, p2 = 'YJL128C', 'YLR113W'

kinases_cor = [empirical_pvalue(p1, p2) for p1, p2 in it.combinations(kinase_df.index, 2)]
kinases_cor = DataFrame(kinases_cor, columns=['p1', 'p2', 'cor', 'a_pvalue', 'e_pvalue', 'n_meas'])
kinases_cor['adj_a_pvalue'] = multipletests(kinases_cor['a_pvalue'], method='fdr_bh')[1]
kinases_cor['adj_e_pvalue'] = multipletests(kinases_cor['e_pvalue'], method='fdr_bh')[1]
kinases_cor.to_csv(wd + 'tables/kinases_pairs_correlation.tab', sep='\t', index=False)
print '[INFO] Kinases correlations done:', len(kinases_cor)