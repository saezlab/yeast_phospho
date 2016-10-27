#!/usr/bin/env python
# Copyright (C) 2016  Emanuel Goncalves

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
from yeast_phospho.utilities import randomise_matrix
from scipy.stats.stats import spearmanr, pearsonr, ttest_ind


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
print 'metabolomics', metabolomics.shape

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
for metabolite, feature, coef, type in assoc[['Metabolites', 'feature', 'coef', 'type']].values:
    if metabolite in metabolomics.index and feature in metabolomics.columns:
        val_df.append({
            'feature': feature, 'metabolite': metabolite, 'coef': coef,
            'coef_binary': 'Negative' if coef < 0 else 'Positive',
            'zscore': metabolomics.ix[metabolite, feature], 'zscore_abs': abs(metabolomics.ix[metabolite, feature]),
            'type': type, 'random': 'No'
        })

        for i in metabolomics_r:
            val_df.append({
                'feature': feature, 'metabolite': metabolite, 'coef': coef,
                'coef_binary': 'Negative' if coef < 0 else 'Positive',
                'zscore': metabolomics_r[i].ix[metabolite, feature], 'zscore_abs': abs(metabolomics_r[i].ix[metabolite, feature]),
                'type': type, 'random': 'Yes'
            })


val_df = DataFrame(val_df)
val_df.to_csv('./tables/validations_external.csv', index=False)
print val_df

# Plot
plot_df = val_df[val_df['coef'].abs() != 0]
print plot_df[plot_df['random'] == 'No'].sort('zscore_abs')

t, pval = ttest_ind(
    plot_df.loc[plot_df['random'] == 'Yes', 'zscore_abs'],
    plot_df.loc[plot_df['random'] == 'No', 'zscore_abs']
)
print t, pval

# Plot
sns.set(style='ticks', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'lines.linewidth': .75})
sns.boxplot('random', 'zscore_abs', data=plot_df, color='#808080', notch=True, fliersize=2)
# sns.stripplot('random', 'zscore_abs', data=plot_df, color='#808080', edgecolor='white', linewidth=.3, jitter=.2)
plt.axhline(0, ls='-', lw=.1, c='gray')
sns.despine()
plt.xlabel('Association')
plt.ylabel('Metabolite (zscore)')
plt.title('Feature knockdown\n(p-value %.2e)' % pval)
plt.gcf().set_size_inches(1.5, 3)
plt.legend(loc=4)
plt.savefig('./reports/associations_metabolomics_cor_boxplots_external.pdf', bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'
