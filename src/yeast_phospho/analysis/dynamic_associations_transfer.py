import re
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools as it
from yeast_phospho import wd
from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_curve, auc
from scipy.stats.distributions import hypergeom
from pandas import DataFrame, Series, read_csv, concat, pivot_table
from yeast_phospho.utilities import get_metabolites_name, get_proteins_name


# -- Import IDs maps
acc_name = get_proteins_name()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}

met_name = get_metabolites_name()
met_name = {'%.4f' % float(k): met_name[k] for k in met_name if len(met_name[k].split('; ')) == 1}


# -- Import reported interactions
with open('%s/tables/protein_metabolite_associations.pickle' % wd, 'rb') as handle:
    interactions = pickle.load(handle)


# -- Import data-sets
# Nitrogen metabolism Metabolomics
metabolomics_dyn_ng = read_csv('%s/tables/metabolomics_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
metabolomics_dyn_ng.index = ['%.4f' % i for i in metabolomics_dyn_ng.index]
metabolomics_dyn_ng = metabolomics_dyn_ng[[i in met_name for i in metabolomics_dyn_ng.index]]
print '[INFO] Nitrogen metabolomics: ', metabolomics_dyn_ng.shape

# Nitrogen metabolism Kinases activities
k_activity_dyn_ng_gsea = read_csv('%s/tables/kinase_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_ng_gsea = k_activity_dyn_ng_gsea[(k_activity_dyn_ng_gsea.count(1) / k_activity_dyn_ng_gsea.shape[1]) > .75].replace(np.NaN, 0.0)
print '[INFO] Nitrogen kinases activities: ', k_activity_dyn_ng_gsea.shape


# Salt+Pheromone Kinases activities
k_activity_dyn_comb_ng = read_csv('%s/tables/kinase_activity_dynamic_combination_gsea.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_comb_ng = k_activity_dyn_comb_ng[[c for c in k_activity_dyn_comb_ng if not c.startswith('NaCl+alpha_')]]
k_activity_dyn_comb_ng = k_activity_dyn_comb_ng[(k_activity_dyn_comb_ng.count(1) / k_activity_dyn_comb_ng.shape[1]) > .75].replace(np.NaN, 0.0)
print '[INFO] Salt+pheromone kinases activities: ', k_activity_dyn_comb_ng.shape

# Salt+Pheromone Metabolomics
metabolomics_dyn_comb = read_csv('%s/tables/metabolomics_dynamic_combination.csv' % wd, index_col=0)[k_activity_dyn_comb_ng.columns]
metabolomics_dyn_comb.index = ['%.4f' % round(i, 2) for i in metabolomics_dyn_comb.index]
metabolomics_dyn_comb = metabolomics_dyn_comb[[i in met_name for i in metabolomics_dyn_comb.index]]
print '[INFO] Salt+pheromone metabolomics: ', metabolomics_dyn_comb.shape


# -- Overlap
ions = list(set(metabolomics_dyn_ng.index).intersection(metabolomics_dyn_comb.index))
kinases = list(set(k_activity_dyn_ng_gsea.index).intersection(k_activity_dyn_comb_ng.index))
conditions = ['alpha', 'NaCl', 'N_downshift', 'N_upshift', 'Rapamycin']

ys = concat([metabolomics_dyn_ng.ix[ions], metabolomics_dyn_comb.ix[ions]], axis=1)
xs = concat([k_activity_dyn_ng_gsea.ix[kinases], k_activity_dyn_comb_ng.ix[kinases]], axis=1)


# -- Linear regressions
lm_res = []
for ion in ions:
    for condition in conditions:
        train, test = [c for c in xs if not re.match(condition, c)], [c for c in xs if re.match(condition, c)]

        ys_train, xs_train = ys.ix[ion, train], sm.add_constant(xs.ix[kinases, train].T)
        ys_test, xs_test = ys.ix[ion, test], sm.add_constant(xs.ix[kinases, test].T)

        lm = sm.OLS(ys_train, xs_train).fit_regularized(L1_wt=.5, alpha=.01)

        pred = Series(lm.predict(xs_test), index=test)
        cor, pval = pearsonr(pred, ys_test.ix[pred.index])

        lm_res.append((ion, condition, cor, pval, lm))

lm_res = DataFrame(lm_res, columns=['ion', 'condition', 'cor', 'pval', 'lm'])
lm_res['condition'] = lm_res['condition'].replace('alpha', 'Pheromone')
print lm_res.sort('cor')


# -- Plot
label_order = ['N_downshift', 'N_upshift', 'Rapamycin', 'NaCl', 'Pheromone']
palette = {'Rapamycin': '#D25A2B', 'N_upshift': '#5EACEC', 'N_downshift': '#4783C7', 'NaCl': '#CC2229', 'Pheromone': '#6FB353'}


# General Linear regression boxplots
sns.set(style='ticks', font_scale=.75, context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.FacetGrid(lm_res, legend_out=True, aspect=1., size=1.5, sharex=True, sharey=False)
g.map(sns.boxplot, 'cor', 'condition', palette=palette, sym='', linewidth=.3, order=label_order, orient='h')
g.map(sns.stripplot, 'cor', 'condition', palette=palette, jitter=True, size=2, split=True, edgecolor='white', linewidth=.3, order=label_order, orient='h')
g.map(plt.axvline, x=0, ls='-', lw=.1, c='gray')
plt.xlim([-1, 1])
g.set_axis_labels('Pearson correlation\n(predicted vs measured)', '')
g.set_titles(row_template='{row_name}')
g.fig.subplots_adjust(wspace=.05, hspace=.2)
sns.despine(trim=True)
plt.savefig('%s/reports/linear_regression_dynamic_transfer_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'


# Top predicted metabolites boxplots
lm_res_top = set(lm_res[(lm_res['pval'] < .05) & (lm_res['cor'] > .0)]['ion'])
lm_res_top = lm_res[[i in lm_res_top for i in lm_res['ion']]]
lm_res_top['name'] = [met_name[i] for i in lm_res_top['ion']]

order = [met_name[i] for i in lm_res_top.groupby('ion')['cor'].median().sort_values(ascending=False).index if len(met_name[i]) < 36]

sns.set(style='ticks', font_scale=.75, context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.FacetGrid(lm_res_top, legend_out=True, aspect=1, size=3, sharex=True, sharey=False)
g.map(sns.boxplot, 'cor', 'name', sym='', orient='h', order=order, color='#CCCCCC', saturation=.1, linewidth=.3)
g.map(sns.stripplot, 'cor', 'name', 'condition', palette=palette, jitter=True, size=2, split=True, edgecolor='white', linewidth=.3, orient='h', order=order, color='#808080')
g.map(plt.axvline, x=0, ls='-', lw=.1, c='gray')
plt.xlim([-1, 1])
g.add_legend(label_order=label_order)
g.set_axis_labels('Pearson correlation\n(predicted vs measured)', '')
g.set_titles(row_template='{row_name}')
g.fig.subplots_adjust(wspace=.05, hspace=.2)
sns.despine(trim=True)
plt.savefig('%s/reports/linear_regression_dynamic_transfer_metabolites_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'


# Important features ROC
lm_res_feat = [(i, f, c, p, lm.params[f], lm.pvalues[f], lm.tvalues[f]) for i, c, p, lm in lm_res[['ion', 'cor', 'pval', 'lm']].values for f in kinases if f != 'const']
lm_res_feat = DataFrame(lm_res_feat, columns=['ion', 'feature', 'cor', 'pval', 'f_coef', 'f_pval', 'f_tstat'])

lm_res_feat['beta (abs)'] = [abs(i) for i in lm_res_feat['f_coef']]
lm_res_feat['t-stat (abs)'] = [abs(i) for i in lm_res_feat['f_tstat']]
lm_res_feat['p-value (-log10)'] = [-np.log10(i) for i in lm_res_feat['f_pval']]

lm_res_feat['biogrid'] = [int((f, i) in interactions['kinases']['biogrid']) for i, f in lm_res_feat[['ion', 'feature']].values]
lm_res_feat['string'] = [int((f, i) in interactions['kinases']['string']) for i, f in lm_res_feat[['ion', 'feature']].values]
lm_res_feat['targets'] = [int((f, i) in interactions['kinases']['targets']) for i, f in lm_res_feat[['ion', 'feature']].values]

lm_res_feat['Kinases/Phosphatases'] = [acc_name[c] for c in lm_res_feat['feature']]
lm_res_feat['Metabolites'] = [met_name[c] for c in lm_res_feat['ion']]


roc_table = lm_res_feat.groupby(['ion', 'feature'])['t-stat (abs)', 'beta (abs)', 'targets', 'biogrid', 'string'].median().reset_index().replace(np.nan, 0)

sns.set(style='ticks', context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
(f, plot), pos = plt.subplots(1, 3, figsize=(3 * 3, 2.5)), 0
for source in ['targets', 'biogrid', 'string']:
    ax = plot[pos]

    for roc_metric in ['t-stat (abs)', 'beta (abs)']:
        curve_fpr, curve_tpr, thresholds = roc_curve(lm_res_feat[source], lm_res_feat[roc_metric])
        curve_auc = auc(curve_fpr, curve_tpr)

        ax.plot(curve_fpr, curve_tpr, label='%s (area = %0.2f)' % (roc_metric, curve_auc))

    ax.plot([0, 1], [0, 1], 'k--', lw=.3)
    ax.set_title(source)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    sns.despine(trim=True, ax=ax)
    ax.legend(loc='lower right')

    pos += 1

plt.savefig('%s/reports/linear_regression_dynamic_transfer_metabolites_rocauc_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'

# Hypergeometric test
# hypergeom.sf(x, M, n, N, loc=0)
# M: total number of objects,
# n: total number of type I objects
# N: total number of type I objects drawn without replacement
db = interactions['kinases']['string']

kinase_enzyme_all = set(it.product(kinases, ions))
kinase_enzyme_true = {i for i in kinase_enzyme_all if i in db}
kinase_enzyme_thres = {(f, m) for f, m, c in lm_res_feat[['feature', 'ion', 't-stat (abs)']].values if c > .1}

pval = hypergeom.sf(
    len(kinase_enzyme_thres.intersection(kinase_enzyme_true)),
    len(kinase_enzyme_all),
    len(kinase_enzyme_all.intersection(kinase_enzyme_true)),
    len(kinase_enzyme_thres)
)
print pval


# Top predicted metabolites features importance
lm_res_top_features = lm_res_feat[[i in order for i in lm_res_feat['Metabolites']]]
lm_res_top_features_matrix = pivot_table(lm_res_top_features, index='Metabolites', columns='Kinases/Phosphatases', values='f_tstat', aggfunc=np.median, fill_value=0)

cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)
sns.set(context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.clustermap(lm_res_top_features_matrix, figsize=(4, 5), linewidth=.5, cmap=cmap, metric='correlation')

for r, c, string, biogrid, target in lm_res_top_features[['Metabolites', 'Kinases/Phosphatases', 'string', 'biogrid', 'targets']].values:
    if c in g.data2d.columns and r in g.data2d.index and (string + biogrid + target) > 0:
        text_x, text_y = (list(g.data2d.columns).index(c), (g.data2d.shape[0] - 1) - list(g.data2d.index).index(r))
        g.ax_heatmap.annotate('*' if (string + biogrid + target) == 1 else '+', (text_x, text_y), xytext=(text_x + .5, text_y + .2), ha='center', va='baseline', color='#808080')

plt.savefig('%s/reports/linear_regression_dynamic_transfer_metabolites_heatmap_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'
