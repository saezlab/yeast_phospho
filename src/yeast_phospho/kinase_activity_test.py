from __future__ import division
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tools as st
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas.stats.misc import zscore
from sklearn.linear_model import Ridge
from pandas import DataFrame, Series, read_csv, melt, pivot_table
from yeast_phospho.estimate_activity import get_kinases_targets


acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=0)
acc_name.index = [i.split(';')[0] for i in acc_name.index]
acc_name = acc_name['oln'].to_dict()

# ---- Import data
# data = read_csv('%s/tables/pproteomics_dynamic.tab' % wd, sep='\t', index_col=0)

data = read_csv('%s/tables/pproteomics_steady_state.tab' % wd, sep='\t', index_col=0)


# ---- Import kinase targets matrix
k_targets_phosphogrid = get_kinases_targets({})
k_targets_phosphogrid_filtered = get_kinases_targets({'19779198', '21177495'})

k_targets = read_csv('%s/tables/similarity_score_df.csv' % wd, sep='\t').dropna()
k_targets['Target'] = [site.split('_')[0] + '_' + seq[7] + site.split('_')[1] for site, seq in k_targets[['Target', 'Sequence']].values]
k_targets['Regulator'] = [acc_name[i] for i in k_targets['Regulator']]
k_targets = pivot_table(k_targets, values='SS', index='Target', columns='Regulator', fill_value=0.0)


for k in k_targets:
    l1 = set(k_targets.loc[k_targets[k] != 0, k].index)
    l2 = set(k_targets_phosphogrid.loc[k_targets_phosphogrid[k] != 0, k].index)

    print k, len(l1.intersection(l2)) / len(l1)


# # ---- Calculate kinase activity with Sklearn
# def k_activity_with_sklearn(x, y):
#     ys = y.ix[x.index].dropna()
#     xs = x.ix[ys.index]
#
#     xs = xs.loc[:, xs.sum() != 0]
#
#     lm = Ridge().fit(xs, zscore(ys))
#
#     return dict(zip(*(xs.columns, lm.coef_)))
#
# k_activity_sklearn = DataFrame({c: k_activity_with_sklearn(k_targets, data[c]) for c in data})


# ---- Calculate kinase activity with Statsmodel
def k_activity_with_statsmodel(x, y):
    ys = y.dropna()
    xs = x.ix[ys.index].replace(np.NaN, 0.0)

    xs = xs.loc[:, xs.sum() != 0]

    lm = sm.OLS(zscore(ys), st.add_constant(xs))

    res = lm.fit_regularized(L1_wt=0)

    print res.summary()

    return res.params.drop('const').to_dict()

k_activity_statsmodel = DataFrame({c: k_activity_with_statsmodel(k_targets, data[c]) for c in data})

# print k_activity_statsmodel[[c for c in k_activity_statsmodel if c.startswith('Rapamycin')]].ix['YJR066W']

# ---- Plot
# ---- Import YeastGenome gene annotation
gene_annotation = read_csv('%s/files/gene_association.txt' % wd, sep='\t', header=None).dropna(how='all', axis=1)
gene_annotation['gene'] = [i.split('|')[0] if str(i) != 'nan' else '' for i in gene_annotation[10]]
gene_annotation = gene_annotation.groupby('gene').first()

kinases_type = DataFrame([(i, gene_annotation.ix[i, 9]) for i in k_activity_statsmodel.index], columns=['name', 'info'])
kinases_type['type'] = ['Kinase' if 'kinase' in i.lower() else 'Phosphatase' if 'phosphatase' in i.lower() else 'ND' for i in kinases_type['info']]
kinases_type = kinases_type.set_index('name')


plot_df = k_activity_statsmodel.copy()
plot_df.columns.name = 'strain'
plot_df['kinase'] = plot_df.index
plot_df = melt(plot_df, id_vars='kinase', value_name='activity')
plot_df['type'] = [kinases_type.ix[i, 'type'] for i in plot_df['kinase']]
plot_df['diagonal'] = ['KO' if k == s else 'WT' for k, s in plot_df[['kinase', 'strain']].values]
plot_df['#targets'] = [len(set(k_targets[k_targets[k] == 1].index).intersection(data.index)) for k in plot_df['kinase']]
plot_df['#targets'] = [str(k) if k <= 10 else '> 10' for k in plot_df['#targets']]


hue_order = ['KO', 'WT']

sns.set(style='ticks', palette='pastel')
g = sns.FacetGrid(data=plot_df, legend_out=True, sharey=False, size=4, aspect=.7)
g.map(plt.axhline, y=0, ls=':', c='.5')
g.map(sns.boxplot, 'type', 'activity', 'diagonal', palette={'KO': '#e74c3c', 'WT': '#95a5a6'}, hue_order=hue_order, sym='', width=.5)
g.map(sns.stripplot, 'type', 'activity', 'diagonal', palette={'KO': '#e74c3c', 'WT': '#95a5a6'}, hue_order=hue_order, jitter=True, size=5)
g.add_legend()
g.set_axis_labels('', 'betas')
sns.despine(trim=True)
g.set_xticklabels(rotation=50)
plt.savefig('%s/reports/kinase_activity_test.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'
