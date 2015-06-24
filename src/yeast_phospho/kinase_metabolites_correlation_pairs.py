from __future__ import division
import itertools as it
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from yeast_phospho import wd
from pandas import DataFrame, Series, read_csv, Index
from scipy.stats.stats import pearsonr


def pearson(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = pearsonr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue, np.sum(mask)

# Import acc map to name form uniprot
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)

# Import kinase activity
kinase_df = read_csv(wd + 'tables/kinase_enrichment_df.tab', sep='\t', index_col=0)

# Import metabolites map
metabol_map = read_csv(wd + 'tables/metabolites_map.tab', sep='\t', index_col=0)
metabol_map.index = Index(metabol_map.index, dtype=str)

# Import metabol log2 FC
metabol_df = read_csv(wd + 'tables/steady_state_metabolomics.tab', sep='\t', index_col=0)
metabol_df.index = Index(metabol_df.index, dtype=str)
metabol_df = metabol_df.ix[set(metabol_map.index).intersection(metabol_df.index)]

# Strains
strains = list(set(kinase_df.columns).intersection(metabol_df.columns))
kinase_df, metabol_df = kinase_df[strains], metabol_df[strains]


# Randomly suffle kinase matrix while keeping NaNs position
def randomise_matrix(matrix):
    random_df = matrix.copy()
    movers = ~np.isnan(random_df.values)
    random_df.values[movers] = np.random.permutation(random_df.values[movers])
    return random_df

n_permutations = 10000
random_kinases = [randomise_matrix(kinase_df) for i in xrange(n_permutations)]
print '[INFO] Kinase activity randomisation done: ', len(random_kinases)


def empirical_pvalue(k, m, verbose=1):
    # Calculate analytical correlaiton p-value
    cor, a_pvalue, n_meas = pearson(kinase_df.ix[k], metabol_df.ix[m])

    # Calculate correlaiton of randomised matrices
    random_cor = [pearson(r_matrix.ix[k], metabol_df.ix[m])[0] for r_matrix in random_kinases]
    count = sum([(r_cor >= cor >= 0) or (r_cor <= cor < 0) for r_cor in random_cor])

    # Calculate empirical p-value
    e_pvalue = 1 / n_permutations if count == 0 else count / n_permutations

    if verbose > 0:
        print '[INFO] ', k, m, cor, a_pvalue, e_pvalue, n_meas

    return k, m, cor, a_pvalue, e_pvalue, n_meas

k_m_cor = [empirical_pvalue(k, m) for k in kinase_df.index for m in metabol_df.index]
k_m_cor = DataFrame(k_m_cor, columns=['k', 'm', 'cor', 'a_pvalue', 'e_pvalue', 'n_meas'])
k_m_cor.to_csv(wd + 'tables/kinases_metabolites_pairs_correlation.tab', sep='\t', index=False)
print '[INFO] Kinases correlations done:', len(k_m_cor)


# Import correlations
k_m_cor = read_csv(wd + 'tables/kinases_metabolites_pairs_correlation.tab', sep='\t').ix[:, range(6)].dropna()
k_m_cor = k_m_cor[k_m_cor['n_meas'] == 115]

k_m_cor['adj_a_pvalue'] = multipletests(k_m_cor['a_pvalue'], method='fdr_bh')[1]
k_m_cor['adj_e_pvalue'] = multipletests(k_m_cor['e_pvalue'], method='fdr_bh')[1]

k_m_cor['k_name'] = [acc_name.ix[i, 'gene'].split(';')[0] for i in k_m_cor['k']]
k_m_cor['m_name'] = [metabol_map.ix[str(i), 'name'] for i in k_m_cor['m']]

k_m_cor[k_m_cor['adj_e_pvalue'] < 0.10]