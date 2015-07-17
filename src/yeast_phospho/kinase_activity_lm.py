import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.regression import mean_squared_error
from yeast_phospho import wd
from scipy.stats.stats import pearsonr, spearmanr
from pandas import DataFrame, Series, read_csv, melt
from sklearn.cross_validation import KFold
from sklearn.linear_model import RidgeCV, Ridge


def spearman(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = spearmanr(x[mask], y[mask])
    return cor, pvalue, np.sum(mask)


def pearson(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = pearsonr(x[mask], y[mask])
    return cor, pvalue, np.sum(mask)

# Import id maps
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)

# Import phospho FC
phospho_df = read_csv(wd + 'tables/steady_state_phosphoproteomics.tab', sep='\t', index_col='site')
strains = list(set(phospho_df.columns))

# Import kinases targets dictionary
k_targets = read_csv(wd + 'tables/kinases_phosphatases_targets.tab', sep='\t')
k_targets = {k: set(k_targets.loc[k_targets['SOURCE'] == k, 'TARGET']) for k in set(k_targets['SOURCE'])}
k_targets = {k: k_targets[k].intersection(phospho_df.index) for k in k_targets}
k_targets = {k: k_targets[k] for k in k_targets if len(k_targets[k]) > 0}
k_targets = DataFrame({k: {t: 1 for t in k_targets[k]} for k in k_targets}).replace(np.NaN, 0)
print '[INFO] [PHOSPHOGRID] Kinases targets: ', k_targets.shape

# ---- Calculate kinase activity


def calculate_activity(strain):
    y = phospho_df.ix[k_targets.index, strain].dropna()
    x = k_targets.ix[y.index]

    x = x.loc[:, x.sum() != 0]

    best_model = (np.Inf, 0.0)
    for train, test in KFold(len(x), 8):
        lm = RidgeCV().fit(x.ix[train], y.ix[train])
        score = mean_squared_error(lm.predict(x.ix[test]), y.ix[test].values)

        if score < best_model[0]:
            best_model = (score, lm.alpha_, lm.coef_)

    print '[INFO] %s, score: %.3f, alpha: %.2f' % (strain, best_model[0], best_model[1])

    return dict(zip(*(x.columns, Ridge(alpha=best_model[0]).fit(x, y).coef_)))

k_activity = DataFrame({c: calculate_activity(c) for c in strains})
print '[INFO] Kinase activity calculated!'

# ---- Plot kinase cluster map
plot_df_order = set(k_activity.index).intersection(k_activity.columns)
plot_df = k_activity.copy().replace(np.NaN, 0).loc[plot_df_order, plot_df_order]
plot_df.index = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.index]
plot_df.columns = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.columns]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'kinases'
sns.clustermap(plot_df, figsize=(20, 20), col_cluster=False, row_cluster=False)
plt.savefig(wd + 'reports/kinase_activity_lm_diagonal.pdf', bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done!'

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

sns.set_style('ticks')
sns.boxplot(x='type', y='value', hue='diagonal', data=plot_df, palette='Paired', orient='v')
sns.stripplot(x='type', y='value', hue='diagonal', data=plot_df, size=8, jitter=True, edgecolor='white', palette='Paired')
sns.despine(offset=10, trim=True)
plt.savefig(wd + 'reports/kinase_activity_lm_diagonal_boxplot.pdf', bbox_inches='tight')
plt.close('all')
