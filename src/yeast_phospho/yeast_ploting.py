import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics.metrics import roc_curve, auc, jaccard_similarity_score
from pandas import DataFrame, Series, read_csv, melt


def pearson(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = pearsonr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue

wd = '/Users/emanuel/Projects/projects/yeast_phospho/'

# Seaborn configurations
sns.set_style('white')

# Version
version = 'v1'
print '[INFO] Version: %s' % version

# Import id maps
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)
metabolites_map = read_csv(wd + 'tables/metabolites_map.tab', sep='\t', index_col=0)

met_map = metabolites_map.copy()
met_map.index = ['%.2f' % c for c in met_map.index]

# Import kinase targets
network = read_csv(wd + 'tables/kinases_phosphatases_targets.tab', sep='\t')
kinases = set(network['SOURCE'])
kinases_targets = {k: set(network.loc[network['SOURCE'] == k, 'TARGET']) for k in kinases}

# Import tables
metabol_df = read_csv(wd + 'tables/metabolomics.tab', sep='\t', index_col=0)

phospho_df = read_csv(wd + 'tables/steady_state_phosphoproteomics.tab', sep='\t', index_col='site')

kinase_df = read_csv(wd + 'tables/kinase_enrichment_df.tab', sep='\t', index_col=0)

cor_df = read_csv(wd + 'tables/met_kin_correlation.tab', sep='\t', index_col=0)
s_distance = read_csv(wd + 'tables/metabolites_distances.tab', sep='\t', index_col=0)

info_table = read_csv(wd + 'tables/information_table.tab', sep='\t')
int_enrichment, dbs = read_csv(wd + 'tables/interactions_enrichment.tab', sep='\t'), ['string', 'phosphogrid']

lm_growth = read_csv(wd + 'tables/lm_growth_prediction.tab', sep='\t')
lm_error = read_csv(wd + 'tables/lm_error.tab', sep='\t', index_col=0)
lm_pred = read_csv(wd + 'tables/lm_predicted.tab', sep='\t', index_col=0)
lm_meas = read_csv(wd + 'tables/lm_measured.tab', sep='\t', index_col=0)
lm_features = read_csv(wd + 'tables/lm_features.tab', sep='\t', index_col=0)

# ---- Plot 

# ---- Plot metabolite cluster map
plot_df = metabol_df.copy().replace(np.NaN, 0)
plot_df.columns = [acc_name.loc[x, 'gene'].split(';')[0] for x in plot_df.columns]
plot_df.index = [metabolites_map.ix[metabolites_map['id'] == i, 'name'].values[0] for i in plot_df.index]
sns.clustermap(plot_df, figsize=(25, 20))
plt.savefig(wd + 'reports/%s_metabolites_clustermap.pdf' % version, bbox_inches='tight')
plt.close('all')

# ---- Plot kinase cluster map
plot_df = kinase_df.copy().replace(np.NaN, 0)
plot_df.index = [acc_name.loc[x, 'gene'].split(';')[0] for x in plot_df.index]
plot_df.columns = [acc_name.loc[x, 'gene'].split(';')[0] for x in plot_df.columns]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'kinases'
sns.clustermap(plot_df, figsize=(30, 17), col_cluster=False, row_cluster=False)
plt.savefig(wd + 'reports/%s_kinase_df_clustermap.pdf' % version, bbox_inches='tight')
plt.close('all')

plot_df_order = set(kinase_df.index).intersection(kinase_df.columns)
plot_df = kinase_df.copy().replace(np.NaN, 0).loc[plot_df_order, plot_df_order]
plot_df.index = [acc_name.loc[x, 'gene'].split(';')[0] for x in plot_df.index]
plot_df.columns = [acc_name.loc[x, 'gene'].split(';')[0] for x in plot_df.columns]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'kinases'
sns.clustermap(plot_df, figsize=(8, 8), col_cluster=False, row_cluster=False)
plt.title('GSEA')
plt.savefig(wd + 'reports/%s_kinase_df_diagonal_clustermap.pdf' % version, bbox_inches='tight')
plt.close('all')

# ---- Plot correlation cluster map
plot_df = cor_df.copy().replace(np.NaN, 0)
plot_df.columns = [acc_name.loc[x, 'gene'].split(';')[0] for x in plot_df.columns]
plot_df.index = [metabolites_map.ix[metabolites_map['id'] == i, 'name'].values[0] for i in plot_df.index]
sns.clustermap(plot_df, figsize=(25, 20))
plt.savefig(wd + 'reports/%s_cordf_clustermap.pdf' % version, bbox_inches='tight')
plt.close('all')

# ---- Distance boxplots
(f, distance_plot), pos = plt.subplots(2, 2, sharey=True, figsize=(15, 10)), 0
for k, df in [('Metabolites', metabol_df), ('Kinases', cor_df)]:
    cor_matrix = df.T.corr('spearman')
    overlap_compounds = set(cor_matrix.index).intersection(s_distance.index)
    pairs_cor = [(c1, c2, cor_matrix.loc[c1, c2], s_distance.loc[c1, c2]) for c1, c2 in it.combinations(overlap_compounds, 2)]
    pairs_cor = DataFrame(pairs_cor, columns=['c1', 'c2', 'correlation', 'distance']).dropna()
    pairs_cor['correlation_abs'] = np.abs(pairs_cor['correlation'])
    pairs_cor['distance_type'] = [str(int(i)) if i < 7 else '>= 7' for i in pairs_cor['distance']]

    x, y = pairs_cor[pairs_cor['distance_type'] == '1'], pairs_cor[pairs_cor['distance_type'] == '>= 7']
    stat, pvalue = ttest_ind(x['correlation_abs'], y['correlation_abs'], equal_var=False)

    # Plotting
    ax = distance_plot[pos][0]
    sns.boxplot(pairs_cor['correlation_abs'], groupby=pairs_cor['distance_type'], ax=ax, color='Set2', alpha=0.8)
    sns.despine(left=True, bottom=True, ax=ax)
    ax.set_ylabel('%s correlations (abs)' % k)

    ax = distance_plot[pos][1]
    pallete = sns.color_palette('Set2', 7)
    sns.boxplot([x['correlation_abs'], y['correlation_abs']], names=['1', '>= 7'], ax=ax, color=[pallete[0], pallete[6]], alpha=0.8)
    sns.despine(left=True, bottom=True, ax=ax)
    ax.set_title('p-value: %.2e' % pvalue)

    pos += 1

plt.savefig(wd + 'reports/%s_distance_associations_mm.pdf' % version, bbox_inches='tight')
plt.close('all')

# ---- Feature importance analysis
features = ['kinase_count', 'pvalue_log', 'linear_kernel_abs', 'polynomial_kernel', 'cosine']

(f, enrichemnt_plot), pos = plt.subplots(len(set(int_enrichment['db'])), len(features) + 1, figsize=(6 * len(features), 8)), 0
for bkg_type in set(int_enrichment['db']):
    inner_pos = 0

    x, y = info_table.ix[:, features], info_table.ix[:, 'class_%s' % bkg_type]

    fs = SelectKBest(chi2, len(features)).fit(x, y)
    fs = DataFrame(fs.scores_, columns=['score'], index=features)

    # Feature importance barplot
    ax = enrichemnt_plot[pos][inner_pos]
    fs.plot(kind='barh', ax=ax)
    ax.set_title(bkg_type)
    sns.despine(left=True, bottom=True, ax=ax)
    ax.grid(False)
    ax.set_ylabel(bkg_type)
    inner_pos += 1

    # Feature importance boxplots
    for feature in features:
        ax = enrichemnt_plot[pos][inner_pos]

        plot_df = DataFrame([info_table['class_%s' % bkg_type], info_table[feature]]).T
        sns.boxplot(plot_df[feature], groupby=plot_df['class_%s' % bkg_type], ax=ax, names=['False', 'True'], alpha=0.8)
        sns.despine(left=True, bottom=True, ax=ax)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(feature)

        inner_pos += 1

    pos += 1

plt.savefig(wd + 'reports/%s_feature_importance.pdf' % version, bbox_inches='tight')
plt.close('all')

# ---- Kinase - enzymes interactions enrichment
(f, enrichemnt_plot), pos = plt.subplots(len(dbs), 3, figsize=(20, (6 * len(dbs)))), 0
for bkg_type in dbs:
    # Ploting
    ax = enrichemnt_plot[pos][0]
    N_thres = 1.8
    values = int_enrichment.loc[np.bitwise_and(int_enrichment['db'] == bkg_type, int_enrichment['thres'] == N_thres)]

    plot_df = values[['M', 'n', 'N', 'x']].T
    plot_df.columns = ['value']
    plot_df['variable'] = ['WO filter', 'WO filter', 'W filter', 'W filter']
    plot_df['type'] = ['all', 'reported', 'all', 'reported']

    sns.barplot('variable', 'value', 'type', data=plot_df, ci=0, x_order=['WO filter', 'W filter'], ax=ax)
    sns.despine(left=True, bottom=True, ax=ax)
    ax.set_ylabel(bkg_type.upper())
    ax.set_xlabel('')
    ax.set_title('M: %d, n: %d, N: %d, x: %d, p-value: %.4f\nFilter: -log10(cor p-value) > %.1f' % (values['M'], values['n'], values['N'], values['x'], values['pvalue'], N_thres))

    # Hypergeometric specific thresold analysis
    ax = enrichemnt_plot[pos][1]
    plot_df = int_enrichment.loc[int_enrichment['db'] == bkg_type, ['thres', 'fraction']].copy()
    sns.barplot('thres', 'fraction', data=plot_df, ax=ax)
    sns.despine(left=True, bottom=True, ax=ax)
    ax.set_xlabel('correlation threshold (-log10 p-value)')
    ax.set_ylabel('% of reported Kinase/Enzyme association')

    ax = enrichemnt_plot[pos][2]
    # ROC plot analysis
    for roc_metric in ['pvalue_log', 'linear_kernel_abs']:
        curve_fpr, curve_tpr, _ = roc_curve(info_table['class_%s' % bkg_type], info_table[roc_metric])
        curve_auc = auc(curve_fpr, curve_tpr)

        ax.plot(curve_fpr, curve_tpr, label='%s (area = %0.2f)' % (roc_metric, curve_auc))

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    sns.despine(left=True, bottom=True, ax=ax)
    ax.legend(loc='lower right')

    pos += 1

plt.savefig(wd + 'reports/%s_kinase_enzyme.pdf' % version, bbox_inches='tight')
plt.close('all')
print '[INFO] Plotting done: ', wd + 'plots/%s_kinase_enzyme.pdf' % version

# ---- Growth prediction plot
sns.boxplot(lm_growth['error'], names=['error'])
sns.despine(left=True, bottom=True)
plt.ylabel('error |predicted - measured|')
plt.title('Growth rate prediction')
plt.gcf().set_size_inches(5, 7)
plt.savefig(wd + 'plots/%s_lm_metabolites_growth_barplot.pdf' % version, bbox_inches='tight')
plt.close('all')

# ---- Correlation plot of the errors
plot_df = lm_error.copy().T
plot_df = plot_df[plot_df < 1].replace(np.NaN, -1)
plot_df.index = [met_map.ix[str(i), 'name'] if str(i) in met_map.index else str(i) for i in plot_df.index]
plot_df.columns = [acc_name.loc[x, 'gene'].split(';')[0] for x in plot_df.columns]
sns.clustermap(plot_df, figsize=(25, 30))
plt.title('fold-change error |predicted - measured|')
plt.savefig(wd + 'reports/%s_lm_error_clustermap.png' % version, bbox_inches='tight')
plt.close('all')

# ---- Plot samples errors
sample_error_df = [(s, pearson(lm_meas.ix[s], lm_pred.ix[s])) for s in lm_meas.index]
sample_error_df = DataFrame([(m, c, p) for m, (c, p) in sample_error_df], columns=['metabolite', 'correlation', 'pvalue'])
sample_error_df = sample_error_df.sort('correlation')
sample_error_df['adjpvalue'] = multipletests(sample_error_df['pvalue'], method='fdr_bh')[1]
sample_error_df['colour'] = ['#3498db' if x < 0.05 else '#95a5a6' for x in sample_error_df['adjpvalue']]
sample_error_df.index = [acc_name.loc[x, 'gene'].split(';')[0] for x in sample_error_df['metabolite']]
print 'Mean correlation: ', sample_error_df['correlation'].mean()

(f, enrichemnt_plot), pos = plt.subplots(1, 2, figsize=(13, 35)), 0

ax = enrichemnt_plot[0]
sample_error_df['correlation'].plot(kind='barh', ax=ax, grid=False, color=sample_error_df['colour'], alpha=.8)
sns.despine(left=True, bottom=True, ax=ax)
ax.set_xlabel('pearson r')
ax.set_ylabel('')
ax.set_title('Mean pearson r: %.3f' % np.mean(sample_error_df['correlation']))

ax = enrichemnt_plot[1]
sns.boxplot(lm_pred.T, vert=False, order=sample_error_df['metabolite'], ax=ax, color=sample_error_df['colour'].values, alpha=.8, names=sample_error_df.index)
sns.despine(left=True, bottom=True, ax=ax)
ax.set_xlabel('fold-change error (|predicted - measured|)')
ax.set_yticks([])
ax.set_ylabel('')

plt.savefig(wd + 'reports/%s_lm_samples_error_boxplot.pdf' % version, bbox_inches='tight')
plt.close('all')

# ---- Plot metabolites errors
met_error_df = [(m, pearson(lm_meas[m], lm_pred[m])) for m in lm_meas.columns]
met_error_df = DataFrame([(m, c, p) for m, (c, p) in met_error_df], columns=['metabolite', 'correlation', 'pvalue'])
met_error_df = met_error_df.sort('correlation')
met_error_df['adjpvalue'] = multipletests(met_error_df['pvalue'], method='fdr_bh')[1]
met_error_df['colour'] = ['#3498db' if x < 0.05 else '#95a5a6' for x in met_error_df['adjpvalue']]
met_error_df.index = [met_map.ix[str(i), 'name'] if str(i) in met_map.index else str(i) for i in met_error_df['metabolite']]
print 'Mean correlation: ', met_error_df['correlation'].mean()

(f, enrichemnt_plot), pos = plt.subplots(1, 2, figsize=(13, 35)), 0

ax = enrichemnt_plot[0]
met_error_df['correlation'].plot(kind='barh', ax=ax, grid=False, color=met_error_df['colour'], alpha=.8)
sns.despine(left=True, bottom=True, ax=ax)
ax.set_xlabel('pearson r')
ax.set_ylabel('')
ax.set_title('Mean pearson r: %.3f' % np.mean(met_error_df['correlation']))

ax = enrichemnt_plot[1]
sns.boxplot(lm_pred, vert=False, order=met_error_df['metabolite'], ax=ax, color=met_error_df['colour'])
sns.despine(left=True, bottom=True, ax=ax)
ax.set_xlabel('fold-change error (|predicted - measured|)')
ax.set_yticks([])
ax.set_ylabel('')

plt.savefig(wd + 'reports/%s_lm_metabolites_error_boxplot.pdf' % version, bbox_inches='tight')
plt.close('all')

# Feature importance
plot_df = lm_features.copy().T
plot_df.index = [met_map.ix[str(i), 'name'] if str(i) in met_map.index else str(i) for i in plot_df.index]
plot_df.columns = [acc_name.loc[x, 'gene'].split(';')[0] for x in plot_df.columns]
sns.clustermap(plot_df, figsize=(25, 30), cmap='Blues')
plt.savefig(wd + 'reports/%s_lm_features_clustermap.pdf' % version, bbox_inches='tight')
plt.close('all')

print '[INFO] Plotting done: ', wd