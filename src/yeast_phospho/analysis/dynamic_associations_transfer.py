import re
import pickle
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as it
from yeast_phospho import wd
from pandas.stats.misc import zscore
from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import ElasticNetCV, ElasticNet, RidgeCV
from sklearn.metrics.regression import r2_score
from scipy.stats.distributions import hypergeom
from sklearn.cross_validation import ShuffleSplit, LeaveOneOut
from pandas import DataFrame, Series, read_csv, concat, pivot_table
from yeast_phospho.utilities import get_metabolites_name, get_proteins_name


# -- General vars
label_order = ['N_downshift', 'N_upshift', 'Rapamycin', 'NaCl', 'Pheromone']
palette = {'Rapamycin': '#D25A2B', 'N_upshift': '#5EACEC', 'N_downshift': '#4783C7', 'NaCl': '#CC2229', 'Pheromone': '#6FB353'}


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

# Replace alpha with pheromone
k_activity_dyn_comb_ng.columns = [c.replace('alpha', 'Pheromone') for c in k_activity_dyn_comb_ng.columns]
metabolomics_dyn_comb.columns = [c.replace('alpha', 'Pheromone') for c in metabolomics_dyn_comb.columns]


# -- Overlap
ions = list(set(metabolomics_dyn_ng.index).intersection(metabolomics_dyn_comb.index))
kinases = list(set(k_activity_dyn_ng_gsea.index).intersection(k_activity_dyn_comb_ng.index))
conditions = ['Pheromone', 'NaCl', 'N_downshift', 'N_upshift', 'Rapamycin']

ys = concat([metabolomics_dyn_ng.ix[ions], metabolomics_dyn_comb.ix[ions]], axis=1)
xs = concat([k_activity_dyn_ng_gsea.ix[kinases], k_activity_dyn_comb_ng.ix[kinases]], axis=1)

# -- Predict experiments
# condition, ion = 'N_downshift', '237.0300'
# condition, ion = 'N_upshift', '188.0600'
lm_res = []
for ion in ions:
    for condition in conditions:
        # Define train and test conditions
        train, test = [c for c in xs if not re.match(condition, c)], [c for c in xs if re.match(condition, c)]

        ys_train, xs_train = ys.ix[ion, train], xs.ix[kinases, train].T
        ys_test, xs_test = ys.ix[ion, test], xs.ix[kinases, test].T

        # Standardization
        xs_train /= xs_train.std()
        xs_test /= xs_test.std()

        ys_train -= ys_train.mean()
        ys_test -= ys_test.mean()

        # Elastic Net ShuffleSplit cross-validation
        cv = ShuffleSplit(len(ys_train), n_iter=20, test_size=.2)
        lm = ElasticNetCV(cv=cv, alphas=np.arange(0, .1, .01)).fit(xs_train, ys_train)

        # Evaluate predictions
        meas, pred = ys_test[test].values, lm.predict(xs_test.ix[test])

        rsquared = r2_score(meas, pred)
        cor, pval = pearsonr(meas, pred)

        # Store results
        lm_res.append((ion, condition, cor, pval, rsquared, lm))

lm_res = DataFrame(lm_res, columns=['ion', 'condition', 'cor', 'pval', 'rsquared', 'lm'])
print lm_res.sort('rsquared')


# Plot General Linear regression boxplots
sns.set(style='ticks', font_scale=.75, context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.FacetGrid(lm_res, legend_out=True, aspect=1., size=1.5, sharex=True, sharey=False)
g.map(sns.boxplot, 'cor', 'condition', palette=palette, sym='', linewidth=.3, order=label_order, orient='h')
g.map(sns.stripplot, 'cor', 'condition', palette=palette, jitter=True, size=2, split=True, edgecolor='white', linewidth=.3, order=label_order, orient='h')
g.map(plt.axvline, x=0, ls='-', lw=.1, c='gray')
plt.xlim([-1, 1])
g.set_axis_labels('Pearson correlation\n(measured ~ predicted)', '')
g.set_titles(row_template='{row_name}')
g.fig.subplots_adjust(wspace=.05, hspace=.2)
sns.despine(trim=True)
plt.savefig('%s/reports/lm_dynamic_boxplots_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'

# Top predicted metabolites boxplots
# lm_res[(lm_res['cor'] > 0) & (lm_res['pval'] < .05)]
lm_res_top = lm_res[(lm_res['rsquared'] > 0) & (lm_res['pval'] < .05)].sort('rsquared', ascending=False)
lm_res_top['name'] = [met_name[i] for i in lm_res_top['ion']]

order = [met_name[i] for i in lm_res_top.groupby('ion')['rsquared'].max().sort_values(ascending=False).index]

sns.set(style='ticks', font_scale=.75, context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.FacetGrid(lm_res_top, legend_out=True, aspect=1, size=3, sharex=True, sharey=False)
g.map(sns.stripplot, 'rsquared', 'name', 'condition', palette=palette, jitter=False, size=4, split=False, edgecolor='white', linewidth=.3, orient='h', order=order, color='#808080')
g.map(plt.axvline, x=0, ls='-', lw=.1, c='gray')
plt.xlim([0, 1])
g.add_legend(label_order=label_order)
g.set_axis_labels('R-squared', '')
g.set_titles(row_template='{row_name}')
g.fig.subplots_adjust(wspace=.05, hspace=.2)
sns.despine(trim=True)
plt.savefig('%s/reports/lm_dynamic_metabolites_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'


# -- Predict associations
# # Filter Pheromone condition
# xs = xs.loc[:, [not c.startswith('Pheromone') for c in xs]]
# ys = ys.loc[:, [not c.startswith('Pheromone') for c in ys]]

# Run Linear models
lm_f_res, n_iter = [], 20
# train, _ = list(ShuffleSplit(len(ys.ix[ion]), test_size=.2))[0]
for ion in ions:
    for train, _ in ShuffleSplit(len(ys.ix[ion]), test_size=.1, n_iter=n_iter):
        ys_train, xs_train = ys.ix[ion, train], xs.ix[kinases, train].T

        # Standardization
        xs_train /= xs_train.std()
        ys_train -= ys_train.mean()

        # Elastic Net ShuffleSplit cross-validation
        cv = ShuffleSplit(len(ys_train), test_size=.1, n_iter=n_iter)
        lm = ElasticNetCV(cv=cv, alphas=np.arange(0, .1, .01)).fit(xs_train, ys_train)

        # Store results
        for f, v in zip(*(kinases, lm.coef_)):
            lm_f_res.append((ion, f, v))

lm_f_res = DataFrame(lm_f_res, columns=['ion', 'feature', 'coef'])
lm_f_res['coef_abs'] = [abs(i) for i in lm_f_res['coef']]

lm_f_res['biogrid'] = [int((f, i) in interactions['kinases']['biogrid']) for i, f in lm_f_res[['ion', 'feature']].values]
lm_f_res['string'] = [int((f, i) in interactions['kinases']['string']) for i, f in lm_f_res[['ion', 'feature']].values]
lm_f_res['targets'] = [int((f, i) in interactions['kinases']['targets']) for i, f in lm_f_res[['ion', 'feature']].values]

lm_f_res['Kinases/Phosphatases'] = [acc_name[c] for c in lm_f_res['feature']]
lm_f_res['Metabolites'] = [met_name[c] for c in lm_f_res['ion']]
print lm_f_res.sort('coef_abs', ascending=False)

# Important features ROC
source_pal = {'string': '#e74c3c', 'biogrid': '#34495e', 'targets': '#2ecc71'}

sns.set(style='ticks', context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
for source in ['targets', 'biogrid', 'string']:
    curve_fpr, curve_tpr, thresholds = roc_curve(lm_f_res[source], lm_f_res['coef_abs'])
    curve_auc = auc(curve_fpr, curve_tpr)

    plt.plot(curve_fpr, curve_tpr, label='%s (area = %0.2f)' % (source, curve_auc), color=source_pal[source])

plt.plot([0, 1], [0, 1], 'k--', lw=.3)
sns.despine(trim=True)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='lower right')

plt.gcf().set_size_inches(3, 3)

plt.savefig('%s/reports/lm_dynamic_roc_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'

# # Hypergeometric test
# # hypergeom.sf(x, M, n, N, loc=0)
# # M: total number of objects,
# # n: total number of type I objects
# # N: total number of type I objects drawn without replacement
# db = interactions['kinases']['string']
#
# kinase_enzyme_all = set(it.product(kinases, ions))
# kinase_enzyme_true = {i for i in kinase_enzyme_all if i in db}
# kinase_enzyme_thres = {(f, m) for f, m, c in lm_f_res[['feature', 'ion', 't-stat (abs)']].values if c > .1}
#
# pval = hypergeom.sf(
#     len(kinase_enzyme_thres.intersection(kinase_enzyme_true)),
#     len(kinase_enzyme_all),
#     len(kinase_enzyme_all.intersection(kinase_enzyme_true)),
#     len(kinase_enzyme_thres)
# )
# print pval

# Top predicted metabolites features importance
lm_res_top_features = lm_f_res[[i in order for i in lm_f_res['Metabolites']]]
t_matrix = pivot_table(lm_res_top_features, index='Metabolites', columns='Kinases/Phosphatases', values='coef', aggfunc=np.mean)

cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)
sns.set(context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})

g = sns.clustermap(t_matrix, figsize=(4, 6), linewidth=.5, cmap=cmap, metric='correlation')

for r, c, string, biogrid, target in lm_res_top_features[['Metabolites', 'Kinases/Phosphatases', 'string', 'biogrid', 'targets']].values:
    if c in g.data2d.columns and r in g.data2d.index and (string + biogrid + target) > 0:
        text_x, text_y = (list(g.data2d.columns).index(c), (g.data2d.shape[0] - 1) - list(g.data2d.index).index(r))
        g.ax_heatmap.annotate('*' if (string + biogrid + target) == 1 else '+', (text_x, text_y), xytext=(text_x + .5, text_y + .2), ha='center', va='baseline', color='#808080')

plt.savefig('%s/reports/lm_dynamic_heatmap_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'
