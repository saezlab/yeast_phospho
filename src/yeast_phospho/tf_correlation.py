import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pandas import DataFrame, Series, read_csv
from statsmodels.stats.multitest import multipletests


def pearson(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = pearsonr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue, np.sum(mask)

wd = '/Users/emanuel/Projects/projects/yeast_phospho/'

# Import TF enrichment
tf_df = read_csv(wd + 'tables/tf_enrichment_df.tab', sep='\t', index_col=0)

# Import kinase enrichment
kinase_df = read_csv(wd + 'tables/kinase_enrichment_df.tab', sep='\t', index_col=0)

# Overlapping strains
strains = set(tf_df.columns).intersection(kinase_df.columns)
tf_df, kinase_df = tf_df.loc[:, strains], kinase_df.loc[:, strains]

# Correlation between TF and kinases
cor_tf_k = [(k, tf, pearson(kinase_df.ix[k, strains], tf_df.ix[tf, strains])) for k in kinase_df.index for tf in tf_df.index]
cor_tf_k = DataFrame([(k, tf, c, p, n) for k, tf, (c, p, n) in cor_tf_k], columns=['kinase', 'tf', 'cor', 'pvalue', 'meas']).dropna()
cor_tf_k = cor_tf_k[cor_tf_k['meas'] > 10]
cor_tf_k['adj_pvalue'] = multipletests(cor_tf_k['pvalue'], method='fdr_bh')[1]
print '[INFO] Correlation done: TF vs Kinases'