#!/usr/bin/env python
# Copyright (C) 2016  Emanuel Goncalves

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
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
val_df = []
for metabolite, feature, coef, type in assoc[['Metabolites', 'feature', 'coef', 'type']].values:
    if metabolite in metabolomics.index and feature in metabolomics.columns:
        metabolite_zscore = metabolomics.ix[metabolite, feature]
        coef_discrete = 'Negative' if coef < 0 else 'Positive'

        res = {'feature': feature, 'metabolite': metabolite, 'coef': coef, 'coef_binary': coef_discrete, 'zscore': metabolite_zscore, 'type': type, 'random': 'No'}
        val_df.append(res)
        print res


val_df = DataFrame(val_df)
val_df.to_csv('./tables/validations_external.csv', index=False)
print val_df

# Plot
plot_df = val_df[val_df['coef'].abs() > .1]
print plot_df

t, pval = ttest_ind(
    plot_df.loc[(plot_df['coef_binary'] == 'Negative'), 'zscore'],
    plot_df.loc[(plot_df['coef_binary'] == 'Positive'), 'zscore']
)
print t, pval

# Plot
sns.set(style='ticks', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'lines.linewidth': .75})
sns.boxplot('coef_binary', 'zscore', data=plot_df, color='#808080', sym='')
sns.stripplot('coef_binary', 'zscore', data=plot_df, color='#808080', edgecolor='white', linewidth=.3, jitter=.2)
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

# # Plot: Corr
# sns.set(style='ticks', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'lines.linewidth': .75})
# g = sns.jointplot(
#     'coef', 'zscore', plot_df, 'reg', color='#808080', marginal_kws={'hist': False, 'rug': True}, space=0, stat_func=pearsonr, xlim=[-.5, .5]
# )
# plt.axhline(0, ls='-', lw=.1, c='gray')
# plt.axvline(0, ls='-', lw=.1, c='gray')
# plt.xlabel('Association coefficient')
# plt.ylabel('Metabolite (zscore)')
# plt.gcf().set_size_inches(3, 3)
# plt.savefig('./reports/associations_metabolomics_cor.pdf', bbox_inches='tight')
# plt.close('all')
# print '[INFO] Plot done'
