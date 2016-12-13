#!/usr/bin/env python
# Copyright (C) 2016  Emanuel Goncalves

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame, Series
from scipy.stats.stats import spearmanr, pearsonr, ttest_ind
from yeast_phospho.utilities import get_metabolites_name
from yeast_phospho.utilities import randomise_matrix


# -- Imports
# Associations
assoc_tf = read_csv('./tables/supp_table3_tf_metabolites.csv')
assoc_tf['type'] = 'Transcription-factor'

assoc_kp = read_csv('./tables/supp_table3_kp_metabolites.csv')
assoc_kp['type'] = 'Kinase/Phosphatase'

assoc = assoc_tf.append(assoc_kp)
assoc['ion'] = ['%.2f' % i for i in assoc['ion']]
print 'assoc', assoc.shape


# -- Metabolomics
met_name = get_metabolites_name()
met_name = {'%.2f' % float(k): met_name[k] for k in met_name if len(met_name[k].split('; ')) == 1}

metabolomics = read_csv('./tables/metabolomics_steady_state.tab', sep='\t', index_col=0)
# metabolomics = metabolomics[metabolomics.std(1) > .4]
metabolomics.index = ['%.2f' % i for i in metabolomics.index]

dup = Series(dict(zip(*(np.unique(metabolomics.index, return_counts=True))))).sort_values()
metabolomics = metabolomics.drop(dup[dup > 1].index, axis=0)
print 'metabolomics', metabolomics.shape


# --
metabolomics_r = {i: randomise_matrix(metabolomics) for i in range(100)}


# --
val_df = []
for ion, feature, coef, type in assoc[['ion', 'feature', 'coef', 'type']].values:
    if ion in metabolomics.index and feature in metabolomics.columns:
        val_df.append({
            'feature': feature, 'ion': ion, 'coef': coef,
            'coef_binary': 'Negative' if coef < 0 else 'Positive',
            'zscore': metabolomics.ix[ion, feature], 'zscore_abs': abs(metabolomics.ix[ion, feature]),
            'type': type, 'random': 'No'
        })

        for i in metabolomics_r:
            val_df.append({
                'feature': feature, 'ion': ion, 'coef': coef,
                'coef_binary': 'Negative' if coef < 0 else 'Positive',
                'zscore': metabolomics_r[i].ix[ion, feature], 'zscore_abs': abs(metabolomics_r[i].ix[ion, feature]),
                'type': type, 'random': 'Yes'
            })


val_df = DataFrame(val_df)
val_df.to_csv('./tables/validations_internal.csv', index=False)
print val_df


# Plot
plot_df = val_df[val_df['coef'].abs() > .1]
print plot_df[plot_df['random'] == 'No'].sort('zscore_abs')

t, pval = ttest_ind(
    plot_df.loc[(plot_df['random'] == 'Yes') & (plot_df['coef_binary'] == 'Negative'), 'zscore_abs'],
    plot_df.loc[(plot_df['random'] == 'No') & (plot_df['coef_binary'] == 'Negative'), 'zscore_abs'],
    equal_var=False
)
print t, pval

# Plot
sns.set(style='ticks', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'lines.linewidth': .75})
sns.boxplot('random', 'zscore_abs', data=plot_df, color='#808080', fliersize=2)
sns.stripplot('random', 'zscore_abs', data=plot_df, color='#808080', edgecolor='white', linewidth=.3, jitter=.2, size=2)
plt.axhline(0, ls='-', lw=.1, c='gray')
sns.despine()
plt.xlabel('Randomisation')
plt.ylabel('Metabolite (abs(zscore))')
plt.title('Feature knockdown\n(p-value %.2e)' % pval)
plt.gcf().set_size_inches(1.5, 3)
plt.legend(loc=4)
plt.savefig('./reports/associations_metabolomics_cor_boxplots_internal.pdf', bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'


sns.set(style='ticks', font_scale=.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'lines.linewidth': .75})
sns.boxplot('coef_binary', 'zscore', 'random', plot_df[plot_df['coef_binary'] == 'Negative'], fliersize=2, color='#808080', hue_order=['Yes', 'No'], linewidth=.3)
# sns.stripplot('coef_binary', 'zscore', 'random', data=plot_df, edgecolor='white', linewidth=.3, jitter=.2, size=2)
plt.axhline(0, ls='-', lw=.1, c='gray')
sns.despine()
plt.xlabel('Association')
plt.ylabel('Metabolite (zscore)')
plt.title('Feature knockdown\n(p-value %.2e)' % pval)
plt.gcf().set_size_inches(1, 3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Randomised')
plt.savefig('./reports/associations_metabolomics_cor_boxplots_internal_types.pdf', bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'
