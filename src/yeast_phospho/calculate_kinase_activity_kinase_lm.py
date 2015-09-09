import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas import DataFrame, read_csv, melt
from pandas.stats.misc import zscore
from sklearn.linear_model import Ridge, LinearRegression


ridge = Ridge(alpha=.1)

# Import growth rates
growth = read_csv(wd + 'files/strain_relative_growth_rate.txt', sep='\t', index_col=0)['relative_growth']

# ---- Steady-state: Calculate kinase activity

# Import phospho FC
phospho_df = read_csv('%s/tables/pproteomics_steady_state.tab' % wd, sep='\t', index_col=0)
strains = list(set(phospho_df.columns).intersection(growth.index))

# Import kinase targets matrix
k_targets = read_csv('%s/tables/targets_kinases.tab' % wd, sep='\t', index_col=0)
k_targets = k_targets.ix[set(k_targets.index).intersection(phospho_df.index)]
k_targets = k_targets.loc[:, k_targets.sum() != 0]

k_activity, k_ntargets = {}, {}
for strain in strains:
    y = phospho_df[strain].dropna()
    x = k_targets.ix[y.index].replace(np.NaN, 0.0)

    x = x.loc[:, x.sum() != 0]

    k_ntargets[strain] = dict(zip(*(x.columns, x.sum())))
    k_activity[strain] = dict(zip(*(x.columns, ridge.fit(x, zscore(y)).coef_)))


k_activity, k_ntargets = DataFrame(k_activity), DataFrame(k_ntargets).replace(np.NaN, 0)
print '[INFO] Kinase activity calculated: ', k_activity.shape

k_activity.to_csv('%s/tables/kinase_activity_steady_state_with_growth.tab' % wd, sep='\t')
print '[INFO] [KINASE ACTIVITY] Exported to'

k_ntargets.to_csv('%s/tables/kinase_activity_steady_state_ntargets.tab' % wd, sep='\t')
print '[INFO] [KINASE # TARGETS] Exported'

# Regress out growth


def regress_out_growth(kinase):
    x, y = growth.ix[strains].values, k_activity.ix[kinase, strains].values

    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))

    x, y = x[mask], y[mask]

    lm = LinearRegression().fit(np.mat(x).T, y)

    y_ = y - lm.coef_[0] * x - lm.intercept_

    return dict(zip(np.array(strains)[mask], y_))


k_activity = DataFrame({kinase: regress_out_growth(kinase) for kinase in k_activity.index}).T.dropna(axis=0, how='all')
print '[INFO] Growth regressed out from the Kinases activity scores: ', k_activity.shape


# Export kinase activity matrix
k_activity_file = '%s/tables/kinase_activity_steady_state.tab' % wd
k_activity.to_csv(k_activity_file, sep='\t')
print '[INFO] [KINASE ACTIVITY] Exported to: %s' % k_activity_file


# ---- Import YeastGenome gene annotation
gene_annotation = read_csv('%s/files/gene_association.txt' % wd, sep='\t', header=None).dropna(how='all', axis=1)
gene_annotation['gene'] = [i.split('|')[0] if str(i) != 'nan' else '' for i in gene_annotation[10]]
gene_annotation = gene_annotation.groupby('gene').first()

kinases_type = DataFrame([(i, gene_annotation.ix[i, 9]) for i in k_activity.index], columns=['name', 'info'])
kinases_type['type'] = ['Kinase' if 'kinase' in i.lower() else 'Phosphatase' if 'phosphatase' in i.lower() else 'ND' for i in kinases_type['info']]
kinases_type = kinases_type.set_index('name')


plot_df = k_activity.copy()
plot_df.columns.name = 'strain'
plot_df['kinase'] = plot_df.index
plot_df = melt(plot_df, id_vars='kinase', value_name='activity').dropna()
plot_df['type'] = [kinases_type.ix[i, 'type'] for i in plot_df['kinase']]
plot_df['diagonal'] = ['KO' if k == s else 'WT' for k, s in plot_df[['kinase', 'strain']].values]
plot_df['#targets'] = [k_targets[k].sum() for k in plot_df['kinase']]
plot_df['#targets'] = [str(k) if k <= 10 else '> 10' for k in plot_df['#targets']]


sns.set(style='ticks', palette='pastel')
col_order, x_order, hue_order = [str(i) if i <= 10 else '> 10' for i in xrange(12) if i not in [0, 7]], ['Kinase', 'Phosphatase', 'ND'], ['KO', 'WT']
g = sns.FacetGrid(data=plot_df, col='#targets', col_order=col_order, col_wrap=5, legend_out=True, sharey=False, size=2, aspect=1)
g.map(plt.axhline, y=0, ls=':', c='.5')
g.map(sns.boxplot, 'type', 'activity', 'diagonal', palette={'KO': '#e74c3c', 'WT': '#95a5a6'}, hue_order=hue_order, sym='', width=.5)
g.map(sns.stripplot, 'type', 'activity', 'diagonal', palette={'KO': '#e74c3c', 'WT': '#95a5a6'}, hue_order=hue_order, jitter=True, size=5)
g.add_legend()
g.set_axis_labels('', 'betas')
g.set_xticklabels(rotation=50)
sns.despine(trim=True)
plt.savefig('%s/reports/kinase_activity_lm_diagonal_boxplot.pdf' % wd, bbox_inches='tight')
plt.close('all')

sns.set(style='ticks', palette='pastel')
g = sns.FacetGrid(data=plot_df, legend_out=True, sharey=False, size=4, aspect=.7)
g.map(plt.axhline, y=0, ls=':', c='.5')
g.map(sns.boxplot, 'type', 'activity', 'diagonal', palette={'KO': '#e74c3c', 'WT': '#95a5a6'}, hue_order=hue_order, sym='', width=.5)
g.map(sns.stripplot, 'type', 'activity', 'diagonal', palette={'KO': '#e74c3c', 'WT': '#95a5a6'}, hue_order=hue_order, jitter=True, size=5)
g.add_legend()
g.set_axis_labels('', 'betas')
sns.despine(trim=True)
g.set_xticklabels(rotation=50)
plt.savefig('%s/reports/kinase_activity_lm_diagonal_boxplot_all.pdf' % wd, bbox_inches='tight')
plt.close('all')


# ---- Dynamic: Calculate kinase activity
# Import phospho FC
phospho_df_dyn = read_csv('%s/tables/pproteomics_dynamic.tab' % wd, sep='\t', index_col=0)
conditions = list(phospho_df_dyn.columns)

# Import kinase targets matrix
k_targets = read_csv('%s/tables/targets_kinases.tab' % wd, sep='\t', index_col=0)
k_targets = k_targets.ix[set(k_targets.index).intersection(phospho_df_dyn.index)]
k_targets = k_targets.loc[:, k_targets.sum() != 0]
k_targets = k_targets[k_targets.sum(1) != 0]


k_activity_dyn, k_dyn_ntargets = {}, {}
for condition in conditions:
    y = phospho_df_dyn.ix[k_targets.index, condition].dropna()
    x = k_targets.ix[y.index]

    x = x.loc[:, x.sum() != 0]

    k_dyn_ntargets[condition] = dict(zip(*(x.columns, x.sum())))
    k_activity_dyn[condition] = dict(zip(*(x.columns, ridge.fit(x, zscore(y)).coef_)))

k_activity_dyn, k_dyn_ntargets = DataFrame(k_activity_dyn), DataFrame(k_dyn_ntargets).replace(np.NaN, 0)
print '[INFO] Kinase activity calculated: ', k_activity_dyn.shape

# Export kinase activity matrix
k_dyn_ntargets.to_csv('%s/tables/kinase_activity_dynamic_ntargets.tab' % wd, sep='\t')
print '[INFO] [KINASE # TARGETS] Exported'

# Export kinase activity matrix
k_activity_dyn.to_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t')
print '[INFO] [KINASE ACTIVITY] Exported'
