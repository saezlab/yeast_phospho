#!/usr/bin/env python
# Copyright (C) 2016  Emanuel Goncalves

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
from scipy.stats.distributions import hypergeom
from sklearn.metrics.classification import f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.metrics.ranking import roc_auc_score
from yeast_phospho.utilities import randomise_matrix
from scipy.stats.stats import spearmanr, pearsonr, ttest_ind, fisher_exact


# -- Imports
# Associations
assoc_tf = read_csv('./tables/supp_table3_tf_metabolites.csv')
assoc_tf['type'] = 'Transcription-factor'

assoc_kp = read_csv('./tables/supp_table3_kp_metabolites.csv')
assoc_kp['type'] = 'Kinase/Phosphatase'

assoc = assoc_tf.append(assoc_kp)
print 'assoc', assoc.shape

# Metabolomics
metabolomics = read_csv('./data/Mulleder_2016_metabolomics_zscores.csv', index_col=0).drop('gene', axis=1).T
metabolomics_p = read_csv('./data/Mulleder_2016_metabolomics_pvalues.csv', index_col=0).drop('gene', axis=1).T
print 'metabolomics', metabolomics.shape

# metabolomics = metabolomics[metabolomics_p < .05]

# --
m_names = {
    'glutamine': 'L-Glutamine',
    'glycine': 'Glycine',
    'asparagine': 'L-Asparagine',
    'proline': 'L-Proline',
    'histidine': 'L-Histidine',

}

metabolomics = metabolomics.ix[m_names.keys()]
metabolomics.index = [m_names[i] for i in metabolomics.index]
print 'metabolomics', metabolomics.shape

# --
metabolomics_r = {i: randomise_matrix(metabolomics) for i in range(100)}

# --
val_df = []
for metabolite, feature, coef, type, cor, fdr in assoc[['Metabolites', 'feature', 'coef', 'type', 'cor', 'fdr']].values:
    if metabolite in metabolomics.index and feature in metabolomics.columns:
        val_df.append({
            'feature': feature, 'metabolite': metabolite, 'coef': coef,
            'coef_binary': 'Negative' if coef < 0 else 'Positive',
            'zscore': metabolomics.ix[metabolite, feature], 'zscore_abs': abs(metabolomics.ix[metabolite, feature]),
            'cor': cor, 'fdr': fdr,
            'type': type
        })

val_df = DataFrame(val_df)
val_df.to_csv('./tables/validations_external.csv', index=False)
print val_df

# -- Plot
well_pred = {
    'Transcription-factor': {'L-Glutamine', 'L-Proline'},
    'Kinase/Phosphatase': {'L-Histidine', 'L-Asparagine', 'L-Proline'}
}

plot_df = val_df[val_df['coef'].abs() > .1]
plot_df = plot_df[[m in well_pred[t] for t, m in plot_df[['type', 'metabolite']].values]]
plot_df['coef_b'] = ['Positive' if i > 0 else 'Negative' for i in plot_df['coef']]
print plot_df.sort('zscore_abs')


t, pval = ttest_ind(
    plot_df.loc[plot_df['coef_b'] == 'Positive', 'zscore'],
    plot_df.loc[plot_df['coef_b'] == 'Negative', 'zscore']
)
print t, pval

# Plot
sns.set(style='ticks', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'lines.linewidth': .75})
sns.boxplot('coef_b', 'zscore', data=plot_df, color='#808080', fliersize=2)
sns.stripplot('coef_b', 'zscore', data=plot_df, color='#808080', edgecolor='white', linewidth=.3, jitter=.2)
plt.axhline(0, ls='-', lw=.1, c='gray')
sns.despine()
plt.xlabel('Association')
plt.ylabel('Metabolite (zscore)')
plt.title('Protein-metabolite interactions\ngene deletion conditions\n\np-value %.2e' % pval)
plt.gcf().set_size_inches(1.5, 3)
plt.legend(loc=4)
plt.savefig('./reports/associations_metabolomics_cor_boxplots.pdf', bbox_inches='tight')
plt.savefig('./reports/associations_metabolomics_cor_boxplots.png', bbox_inches='tight', dpi=300)
plt.close('all')
print '[INFO] Plot done'
