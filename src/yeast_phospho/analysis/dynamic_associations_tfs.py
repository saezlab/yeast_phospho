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
from sklearn.linear_model import ElasticNetCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics.regression import r2_score
from pandas import DataFrame, Series, read_csv, concat, pivot_table
from yeast_phospho.utilities import get_metabolites_name, get_proteins_name


# -- Import IDs maps
acc_name = get_proteins_name()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}

met_name = get_metabolites_name()
met_name = {'%.4f' % float(k): met_name[k] for k in met_name if len(met_name[k].split('; ')) == 1 and met_name[k] != 'NADP'}


# -- Import associations
with open('%s/tables/protein_metabolite_associations.pickle' % wd, 'rb') as handle:
    interactions = pickle.load(handle)


# -- Import data-sets
# Metabolomics
ys = read_csv('%s/tables/metabolomics_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
ys.index = ['%.4f' % i for i in ys.index]
ys = ys[[i in met_name for i in ys.index]]

# GSEA
xs = read_csv('%s/tables/tf_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
xs = xs[xs.std(1) > .4]

conditions, tfs, ions = ['N_downshift', 'N_upshift', 'Rapamycin'], list(xs.index), list(ys.index)


# -- Standardization
xs = xs.divide(xs.std()).T


# -- Linear regressions
lm_res = []
for ion in ions:
    for condition in conditions:
        train, test = [c for c in xs.index if not re.match(condition, c)], [c for c in xs.index if re.match(condition, c)]

        # First ElasticNet
        ys_train, xs_train = ys.ix[ion, train], sm.add_constant(xs.ix[train])

        cv = ShuffleSplit(len(ys_train), n_iter=10)
        lm_a = ElasticNetCV(fit_intercept=False, alphas=[.01, .001], cv=cv).fit(xs_train, ys_train)

        top_features = Series(dict(zip(*(xs_train.columns, lm_a.coef_)))).drop('const')
        top_features = top_features[top_features != 0].abs().sort_values(ascending=False).head(10).index

        # Second Linear regression
        ys_train, xs_train = ys.ix[ion, train], sm.add_constant(xs.ix[train, top_features])
        ys_test, xs_test = ys.ix[ion, test], sm.add_constant(xs.ix[test, top_features])

        lm = sm.OLS(ys_train, xs_train).fit_regularized(L1_wt=lm_a.l1_ratio_, alpha=lm_a.alpha_)

        meas, pred = ys_test[test].values, lm.predict(xs_test.ix[test])

        rsquared = r2_score(meas, pred)

        lm_res.append((ion, condition, rsquared, lm, lm_a.alpha_, lm_a.l1_ratio_))

lm_res = DataFrame(lm_res, columns=['ion', 'condition', 'rsquared', 'lm', 'alpha', 'l1_ratio'])
print lm_res.sort('rsquared')


# -- Plot
label_order = ['N_downshift', 'N_upshift', 'Rapamycin']
palette = {'Rapamycin': '#D25A2B', 'N_upshift': '#5EACEC', 'N_downshift': '#4783C7', 'NaCl': '#CC2229', 'Pheromone': '#6FB353'}

# General Linear regression boxplots
sns.set(style='ticks', font_scale=.75, context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.FacetGrid(lm_res, legend_out=True, aspect=1., size=1.5, sharex=True, sharey=False)
g.map(sns.boxplot, 'rsquared', 'condition', palette=palette, sym='', linewidth=.3, order=label_order, orient='h')
g.map(sns.stripplot, 'rsquared', 'condition', palette=palette, jitter=True, size=2, split=True, edgecolor='white', linewidth=.3, order=label_order, orient='h')
g.map(plt.axvline, x=0, ls='-', lw=.1, c='gray')
g.set_axis_labels('R-squared', '')
g.set_titles(row_template='{row_name}')
g.fig.subplots_adjust(wspace=.05, hspace=.2)
sns.despine(trim=True)
plt.savefig('%s/reports/linear_regression_dynamic_transfer_gsea_tfs.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'


# Top predicted metabolites boxplots
lm_res_top = lm_res[lm_res['rsquared'] > 0]
lm_res_top['name'] = [met_name[i] for i in lm_res_top['ion']]

order = [met_name[i] for i in lm_res_top.groupby('ion')['rsquared'].max().sort_values(ascending=False).index]

sns.set(style='ticks', font_scale=.75, context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.FacetGrid(lm_res_top, legend_out=True, aspect=1.5, size=4, sharex=True, sharey=False)
g.map(sns.stripplot, 'rsquared', 'name', 'condition', palette=palette, jitter=True, size=4, split=True, edgecolor='white', linewidth=.3, orient='h', order=order, color='#808080')
g.map(plt.axvline, x=0, ls='-', lw=.1, c='gray')
plt.xlim([0, 1])
g.add_legend(label_order=label_order)
g.set_axis_labels('R-squared', '')
g.set_titles(row_template='{row_name}')
g.fig.subplots_adjust(wspace=.05, hspace=.2)
sns.despine(trim=True)
plt.savefig('%s/reports/linear_regression_dynamic_transfer_metabolites_gsea_tfs.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'


# Important features ROC
lm_res_feat = [(i, f, c, lm.params[f], lm.pvalues[f], lm.tvalues[f]) if f in lm.params else (i, f, c, 0, np.nan, np.nan) for i, c, lm in lm_res[['ion', 'condition', 'lm']].values for f in tfs if f != 'const']
lm_res_feat = DataFrame(lm_res_feat, columns=['ion', 'feature', 'condition', 'f_coef', 'f_pval', 'f_tstat'])

lm_res_feat['f_pval'] = lm_res_feat['f_pval'].replace(np.nan, 1)
lm_res_feat['f_tstat'] = lm_res_feat['f_tstat'].replace(np.nan, 0)

lm_res_feat['beta (abs)'] = [abs(i) for i in lm_res_feat['f_coef']]
lm_res_feat['t-stat (abs)'] = [abs(i) for i in lm_res_feat['f_tstat']]
lm_res_feat['p-value (-log10)'] = [-np.log10(i) for i in lm_res_feat['f_pval']]

lm_res_feat['biogrid'] = [int((f, i) in interactions['tfs']['biogrid']) for i, f in lm_res_feat[['ion', 'feature']].values]
lm_res_feat['string'] = [int((f, i) in interactions['tfs']['string']) for i, f in lm_res_feat[['ion', 'feature']].values]
lm_res_feat['targets'] = [int((f, i) in interactions['tfs']['targets']) for i, f in lm_res_feat[['ion', 'feature']].values]

lm_res_feat['Transcription-factors'] = [acc_name[c] for c in lm_res_feat['feature']]
lm_res_feat['Metabolites'] = [met_name[c] for c in lm_res_feat['ion']]


source_pal = {'string': '#e74c3c', 'biogrid': '#34495e', 'targets': '#2ecc71'}

sns.set(style='ticks', context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
for source in ['targets', 'biogrid', 'string']:
    curve_fpr, curve_tpr, thresholds = roc_curve(lm_res_feat[source], 1 - lm_res_feat['f_pval'])
    curve_auc = auc(curve_fpr, curve_tpr)

    plt.plot(curve_fpr, curve_tpr, label='%s (area = %0.2f)' % (source, curve_auc), color=source_pal[source])

plt.plot([0, 1], [0, 1], 'k--', lw=.3)
sns.despine(trim=True)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='lower right')

plt.gcf().set_size_inches(3, 3)

plt.savefig('%s/reports/linear_regression_dynamic_transfer_metabolites_rocauc_gsea_tfs.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'

# # Hypergeometric test
# # hypergeom.sf(x, M, n, N, loc=0)
# # M: total number of objects,
# # n: total number of type I objects
# # N: total number of type I objects drawn without replacement
# db = interactions['tfs']['string']
#
# kinase_enzyme_all = set(it.product(tfs, ions))
# kinase_enzyme_true = {i for i in kinase_enzyme_all if i in db}
# kinase_enzyme_thres = {(f, m) for f, m, c in lm_res_feat[['feature', 'ion', 't-stat (abs)']].values if c > .0}
#
# pval = hypergeom.sf(
#     len(kinase_enzyme_thres.intersection(kinase_enzyme_true)),
#     len(kinase_enzyme_all),
#     len(kinase_enzyme_all.intersection(kinase_enzyme_true)),
#     len(kinase_enzyme_thres)
# )
# print pval


# Top predicted metabolites features importance
lm_res_top_features = lm_res_feat[[i in order for i in lm_res_feat['Metabolites']]]
t_matrix = pivot_table(lm_res_top_features, index='Metabolites', columns='Transcription-factors', values='f_tstat', aggfunc=np.mean)
t_matrix = t_matrix.loc[:, t_matrix.std() > .1]

cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)
sns.set(context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.clustermap(t_matrix.replace(np.nan, 0), figsize=(4, 5), linewidth=.5, cmap=cmap, metric='correlation', mask=t_matrix.applymap(np.isnan))

for r, c, string, biogrid, target in lm_res_top_features[['Metabolites', 'Transcription-factors', 'string', 'biogrid', 'targets']].values:
    if c in g.data2d.columns and r in g.data2d.index and (string + biogrid + target) > 0:
        text_x, text_y = (list(g.data2d.columns).index(c), (g.data2d.shape[0] - 1) - list(g.data2d.index).index(r))
        g.ax_heatmap.annotate('*' if (string + biogrid + target) == 1 else '+', (text_x, text_y), xytext=(text_x + .5, text_y + .2), ha='center', va='baseline', color='#808080')

plt.savefig('%s/reports/linear_regression_dynamic_transfer_metabolites_heatmap_gsea_tfs.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'
