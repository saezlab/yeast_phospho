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


# -- Import associations
with open('%s/tables/protein_metabolite_associations_direct_targets.pickle' % wd, 'rb') as handle:
    dbs_direct = pickle.load(handle)

with open('%s/tables/protein_metabolite_associations_protein_interactions.pickle' % wd, 'rb') as handle:
    dbs_associations = pickle.load(handle)


# -- Import data-sets
# Metabolomics
ys = read_csv('%s/tables/metabolomics_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
ys.index = ['%.4f' % i for i in ys.index]

# GSEA
xs = read_csv('%s/tables/tf_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
xs = xs[xs.std(1) > .4]

conditions, tfs = ['N_downshift', 'N_upshift', 'Rapamycin'], list(xs.index)


# -- Linear regressions
lm_res = []
for ion in ys.index:
    for condition in conditions:
        train, test = [c for c in xs if not re.match(condition, c)], [c for c in xs if re.match(condition, c)]

        lm = sm.OLS(ys.ix[ion, train], sm.add_constant(xs.ix[tfs, train].T)).fit_regularized(L1_wt=.5, alpha=.01)

        pred, meas = Series(lm.predict(sm.add_constant(xs.ix[tfs, test].T)), index=test), ys.ix[ion, test]
        cor, pval = pearsonr(pred, meas.ix[pred.index])

        lm_res.append((ion, condition, cor, pval, lm))

lm_res = DataFrame(lm_res, columns=['ion', 'condition', 'cor', 'pval', 'lm'])
lm_res['condition'] = lm_res['condition'].replace('alpha', 'Pheromone')
lm_res['condition'] = lm_res['condition'].replace('.', 'All')
print lm_res.sort('cor')


# -- Plot
label_order = ['N_downshift', 'N_upshift', 'Rapamycin']
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
plt.savefig('%s/reports/linear_regression_dynamic_transfer_gsea_tfs.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'


# Top predicted metabolites boxplots
lm_res_top = {k for k, cor in lm_res.groupby('ion')['cor'].median().to_dict().items() if cor > .5}.intersection({k for k, cor_min in lm_res.groupby('ion')['cor'].min().to_dict().items() if cor_min > .0})
lm_res_top = set(lm_res[([i in lm_res_top for i in lm_res['ion']]) & (lm_res['pval'] < .05)]['ion'])
lm_res_top = lm_res[[i in lm_res_top for i in lm_res['ion']]]
lm_res_top['name'] = [met_name[i] for i in lm_res_top['ion']]

order = [met_name[i] for i in lm_res_top.groupby('ion')['cor'].median().sort_values(ascending=False).index if len(met_name[i]) < 36]

sns.set(style='ticks', font_scale=.75, context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.FacetGrid(lm_res_top, legend_out=True, aspect=1, size=4, sharex=True, sharey=False)
g.map(sns.boxplot, 'cor', 'name', sym='', orient='h', order=order, color='#CCCCCC', saturation=.1, linewidth=.3)
g.map(sns.stripplot, 'cor', 'name', 'condition', palette=palette, jitter=True, size=2, split=True, edgecolor='white', linewidth=.3, orient='h', order=order, color='#808080')
g.map(plt.axvline, x=0, ls='-', lw=.1, c='gray')
plt.xlim([0, 1])
g.add_legend(label_order=label_order)
g.set_axis_labels('Pearson correlation\n(predicted vs measured)', '')
g.set_titles(row_template='{row_name}')
g.fig.subplots_adjust(wspace=.05, hspace=.2)
sns.despine(trim=True)
plt.savefig('%s/reports/linear_regression_dynamic_transfer_metabolites_gsea_tfs.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'


# Top predicted metabolites features importance
lm_res_top_features = DataFrame([(ion, f, c) for ion, lm, cor in lm_res_top[['ion', 'lm', 'cor']].values if cor > 0 and met_name[ion] in order for f, c in lm.params.to_dict().items() if f != 'const'], columns=['i', 'k', 'coef'])
lm_res_top_features['Transcription-factors'] = [acc_name[c] for c in lm_res_top_features['k']]
lm_res_top_features['Metabolites'] = [met_name[c] for c in lm_res_top_features['i']]
lm_res_top_features['reported'] = [int((k, m) in dbs_associations['tfs']) for k, m in lm_res_top_features[['k', 'i']].values]
lm_res_top_features = lm_res_top_features.sort(['reported', 'coef'], ascending=False)

plot_df = pivot_table(lm_res_top_features, index='Metabolites', columns='Transcription-factors', values='coef', aggfunc=np.median)
plot_df = plot_df.loc[:, plot_df.abs().sum() > .1]

cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)
sns.set(context='paper', font_scale=.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.clustermap(plot_df, figsize=(5, 5), linewidth=.5, cmap=cmap, metric='correlation')

for r, c, reported in lm_res_top_features[['Metabolites', 'Transcription-factors', 'reported']].values:
    if c in g.data2d.columns and r in g.data2d.index and reported > 0:
        text_x, text_y = (list(g.data2d.columns).index(c), (g.data2d.shape[0] - 1) - list(g.data2d.index).index(r))
        g.ax_heatmap.annotate('*' if reported == 1 else '+', (text_x, text_y), xytext=(text_x + .5, text_y + .2), ha='center', va='baseline', color='#808080')

plt.savefig('%s/reports/linear_regression_dynamic_transfer_metabolites_heatmap_gsea_tfs.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'


#
lm_res_features = [(i, f, c, p, v, abs(v)) for i, c, p, lm in lm_res[['ion', 'cor', 'pval', 'lm']].values for f, v in lm.params.to_dict().items() if f != 'const']
lm_res_features = DataFrame(lm_res_features, columns=['ion', 'feature', 'cor', 'pval', 'coef', 'abs_coef']).sort('abs_coef', ascending=False)
lm_res_features['reported'] = [int((f, i) in dbs_direct['tfs'] or (f, i) in dbs_associations['tfs']) for f, i in lm_res_features[['feature', 'ion']].values]

curve_fpr, curve_tpr, thresholds = roc_curve(lm_res_features['reported'], lm_res_features['abs_coef'])
curve_auc = auc(curve_fpr, curve_tpr)
print curve_auc

# plt.plot(curve_fpr, curve_tpr, label='Beta (area = %0.2f)' % curve_auc)
#
# plt.plot([0, 1], [0, 1], 'k--')
# plt.despine(trim=True)
# plt.legend(loc='lower right')

# Hypergeometric test
# hypergeom.sf(x, M, n, N, loc=0)
# M: total number of objects,
# n: total number of type I objects
# N: total number of type I objects drawn without replacement
ion_all = {'%.4f' % i for i in read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0).index}
tfs_all = set(read_csv('%s/tables/tf_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0).index)

db = dbs_direct['tfs'].union(dbs_associations['tfs'])

interactions_all = set(it.product(tfs_all, ion_all))
interactions_reported = {i for i in interactions_all if i in db}
interactions_predicted = {(f, m) for f, m, c, coef in lm_res_features[['feature', 'ion', 'cor', 'abs_coef']].values if coef > .1}

pval = hypergeom.sf(
    len(interactions_predicted.intersection(interactions_reported)),
    len(interactions_all),
    len(interactions_all.intersection(interactions_reported)),
    len(interactions_predicted)
)
print pval
