import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model.base import LinearRegression
from sklearn.metrics.regression import mean_squared_error
from yeast_phospho import wd
from pandas import DataFrame, read_csv, melt
from sklearn.cross_validation import KFold
from sklearn.linear_model import RidgeCV, Ridge

sns.set_style('ticks')

# Import id maps
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)

# Import growth rates
growth = read_csv(wd + 'files/strain_relative_growth_rate.txt', sep='\t', index_col=0)['relative_growth']

# ---- Steady-state: Calculate kinase activity

# Import phospho FC
phospho_df = read_csv('%s/tables/pproteomics_steady_state.tab' % wd, sep='\t', index_col=0)
strains = list(set(phospho_df.columns))

# Import kinase targets matrix
k_targets = read_csv('%s/tables/kinases_targets_phosphogrid.tab' % wd, sep='\t', index_col=0)
k_targets = k_targets.ix[set(k_targets.index).intersection(phospho_df.index)]
k_targets = k_targets.loc[:, k_targets.sum() != 0]


def calculate_activity(strain):
    y = phospho_df.ix[k_targets.index, strain].dropna()
    x = k_targets.ix[y.index]

    x = x.loc[:, x.sum() != 0]

    best_model = (np.Inf, 0.0)
    for train, test in KFold(len(x), 3):
        lm = RidgeCV().fit(x.ix[train], y.ix[train])
        score = mean_squared_error(lm.predict(x.ix[test]), y.ix[test].values)

        if score < best_model[0]:
            best_model = (score, lm.alpha_, lm.coef_)

    print '[INFO] %s, score: %.3f, alpha: %.2f' % (strain, best_model[0], best_model[1])

    return dict(zip(*(x.columns, Ridge(alpha=best_model[0]).fit(x, y).coef_)))

k_activity = DataFrame({c: calculate_activity(c) for c in strains})
print '[INFO] Kinase activity calculated: ', k_activity.shape

# Regress out growth


def regress_out_growth(kinase):
    x, y = growth.ix[strains].values, k_activity.ix[kinase, strains].values

    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))

    if sum(mask) > 3:
        x, y = x[mask], y[mask]

        lm = LinearRegression().fit(np.mat(x).T, y)

        y_ = y - lm.coef_[0] * x - lm.intercept_

        return dict(zip(np.array(strains)[mask], y_))

    else:
        return {}

k_activity = DataFrame({kinase: regress_out_growth(kinase) for kinase in k_activity.index}).T.dropna(axis=0, how='all')
print '[INFO] Growth regressed out from the Kinases activity scores: ', k_activity.shape

# Export kinase activity matrix
k_activity_file = '%s/tables/kinase_activity_steady_state.tab' % wd
k_activity.to_csv(k_activity_file, sep='\t')
print '[INFO] [KINASE ACTIVITY] Exported to: %s' % k_activity_file

# ---- Plot kinase cluster map
plot_df_order = set(k_activity.index).intersection(k_activity.columns)
plot_df = k_activity.copy().replace(np.NaN, 0).loc[plot_df_order, plot_df_order]
plot_df.index = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.index]
plot_df.columns = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.columns]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'kinases'
sns.clustermap(plot_df, figsize=(20, 20), col_cluster=False, row_cluster=False)
plt.savefig(wd + 'reports/kinase_activity_lm_diagonal.pdf', bbox_inches='tight')
plt.close('all')
print '[INFO] Clustemap done!'

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
plot_df = melt(plot_df, id_vars='kinase')
plot_df['type'] = [kinases_type.ix[i, 'type'] for i in plot_df['kinase']]
plot_df['diagonal'] = ['Diagonal' if k == s else 'Off-diagonal' for k, s in plot_df[['kinase', 'strain']].values]

sns.boxplot(x='type', y='value', hue='diagonal', data=plot_df, palette='Paired', orient='v')
sns.stripplot(x='type', y='value', hue='diagonal', data=plot_df, size=8, jitter=True, edgecolor='white', palette='Paired')
sns.despine(offset=10, trim=True)
plt.savefig(wd + 'reports/kinase_activity_lm_diagonal_boxplot.pdf', bbox_inches='tight')
plt.close('all')


# ---- Dynamic: Calculate kinase activity
# Import phospho FC
phospho_df_dyn = read_csv('%s/tables/pproteomics_dynamic.tab' % wd, sep='\t', index_col=0)
conditions = list(phospho_df_dyn.columns)

# Import kinase targets matrix
k_targets = read_csv('%s/tables/kinases_targets_phosphogrid.tab' % wd, sep='\t', index_col=0)
k_targets = k_targets.ix[set(k_targets.index).intersection(phospho_df_dyn.index)]
k_targets = k_targets.loc[:, k_targets.sum() != 0]


def calculate_activity_dynamic(condition):
    y = phospho_df_dyn.ix[k_targets.index, condition].dropna()
    x = k_targets.ix[y.index]

    x = x.loc[:, x.sum() != 0]

    best_model = (np.Inf, 0.0)
    for train, test in KFold(len(x), 3):
        lm = RidgeCV().fit(x.ix[train], y.ix[train])
        score = mean_squared_error(lm.predict(x.ix[test]), y.ix[test].values)

        if score < best_model[0]:
            best_model = (score, lm.alpha_, lm.coef_)

    print '[INFO] %s, score: %.3f, alpha: %.2f' % (condition, best_model[0], best_model[1])

    return dict(zip(*(x.columns, Ridge(alpha=best_model[0]).fit(x, y).coef_)))

k_activity_dyn = DataFrame({c: calculate_activity_dynamic(c) for c in conditions})
print '[INFO] Kinase activity calculated: ', k_activity_dyn.shape

# Export kinase activity matrix
k_activity_dyn_file = '%s/tables/kinase_activity_dynamic.tab' % wd
k_activity_dyn.to_csv(k_activity_dyn_file, sep='\t')
print '[INFO] [KINASE ACTIVITY] Exported to: %s' % k_activity_dyn_file
