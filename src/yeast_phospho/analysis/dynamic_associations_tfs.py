import re
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as it
from yeast_phospho import wd
from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_curve, auc
from scipy.stats.distributions import hypergeom
from sklearn.decomposition.pca import PCA
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.cross_validation import ShuffleSplit, LeaveOneOut
from sklearn.metrics.regression import r2_score
from pandas import DataFrame, Series, read_csv, concat, pivot_table
from yeast_phospho.utilities import get_metabolites_name, get_proteins_name, regress_out


# -- General vars
label_order = ['N_downshift', 'N_upshift', 'Rapamycin']
palette = {'Rapamycin': '#D25A2B', 'N_upshift': '#5EACEC', 'N_downshift': '#4783C7'}


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
xs = xs[xs.std(1) > .1]

conditions, tfs, ions = ['N_downshift', 'N_upshift', 'Rapamycin'], list(xs.index), list(ys.index)


n_components = 10
xs_pca = PCA(n_components=n_components).fit(xs.T)
xs_pcs = DataFrame(xs_pca.transform(xs.T), columns=['PC%d' % i for i in range(1, n_components + 1)], index=xs.columns)
print xs_pca.explained_variance_ratio_
xs = DataFrame({i: regress_out(xs_pcs['PC1'], xs.ix[i, xs_pcs.index]) for i in xs.index}).T


ys_pca = PCA(n_components=n_components).fit(ys.T)
ys_pcs = DataFrame(ys_pca.transform(ys.T), columns=['PC%d' % i for i in range(1, n_components + 1)], index=ys.columns)
print ys_pca.explained_variance_ratio_
ys = DataFrame({i: regress_out(ys_pcs['PC1'], ys.ix[i, ys_pcs.index]) for i in ys.index}).T


plot_df = xs.T.corr()
plot_df.index = [acc_name[i] for i in plot_df.index]
plot_df.columns = [acc_name[i] for i in plot_df.columns]

cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)
sns.set(context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.clustermap(plot_df, figsize=(10, 10), linewidth=.5, cmap=cmap, metric='correlation')
plt.title('Nitrogen metabolism\n(pearson)')
plt.savefig('%s/reports/clustermap_nitrogen_tfactivities_with_pc1_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')


plot_df = ys.T.corr()
plot_df.index = [met_name[i] for i in plot_df.index]
plot_df.columns = [met_name[i] for i in plot_df.columns]

cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)
sns.set(context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.clustermap(plot_df, figsize=(15, 15), linewidth=.5, cmap=cmap, metric='correlation')
plt.title('Nitrogen metabolism\n(pearson)')
plt.savefig('%s/reports/clustermap_nitrogen_metabolomics_with_pc1_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')


sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.jointplot(
    xs_pcs['PC1'], ys_pcs['PC1'], kind='reg', color='#808080', joint_kws={'scatter_kws': {'s': 40, 'edgecolor': 'w', 'linewidth': .5}},
    marginal_kws={'hist': False, 'rug': True}, space=0
)
plt.axhline(0, ls='-', lw=0.3, c='#808080', alpha=.5)
plt.axvline(0, ls='-', lw=0.3, c='#808080', alpha=.5)
g.plot_marginals(sns.kdeplot, shade=True, color='#808080')
g.set_axis_labels('TF activities PC1', 'Metabolomics activities PC1')
plt.savefig('%s/reports/pca_correlation.pdf' % wd, bbox_inches='tight')
plt.close('all')


# -- Predict experiments
# condition, ion = 'N_upshift', '188.0600'
lm_res = []
for ion in ions:
    for condition in conditions:
        # Define train and test conditions
        train, test = [c for c in xs if not re.match(condition, c)], [c for c in xs if re.match(condition, c)]

        ys_train, xs_train = ys.ix[ion, train], xs.ix[tfs, train].T
        ys_test, xs_test = ys.ix[ion, test], xs.ix[tfs, test].T

        # Standardization
        xs_train /= xs_train.std()
        xs_test /= xs_test.std()

        ys_train -= ys_train.mean()
        ys_test -= ys_test.mean()

        # # Elastic Net ShuffleSplit cross-validation
        cv = ShuffleSplit(len(ys_train), n_iter=10, test_size=.1)
        lm = ElasticNetCV(cv=cv).fit(xs_train, ys_train)

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
plt.savefig('%s/reports/lm_dynamic_boxplots_tfs_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'


# Top predicted metabolites boxplots
# lm_res[(lm_res['cor'] > 0) & (lm_res['pval'] < .05)]
lm_res_top = lm_res[(lm_res['rsquared'] > 0) & (lm_res['pval'] < .05)].sort('rsquared', ascending=False)
lm_res_top['name'] = [met_name[i] for i in lm_res_top['ion']]

order = [met_name[i] for i in lm_res_top.groupby('ion')['rsquared'].max().sort_values(ascending=False).index]

sns.set(style='ticks', font_scale=.75, context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.FacetGrid(lm_res_top, legend_out=True, aspect=1.5, size=3, sharex=True, sharey=False)
g.map(sns.stripplot, 'rsquared', 'name', 'condition', palette=palette, jitter=False, size=4, split=False, edgecolor='white', linewidth=.3, orient='h', order=order, color='#808080')
plt.xlim([0, 1])
g.add_legend(label_order=label_order)
g.set_axis_labels('R-squared', '')
g.set_titles(row_template='{row_name}')
g.fig.subplots_adjust(wspace=.05, hspace=.2)
sns.despine(trim=True)
plt.savefig('%s/reports/lm_dynamic_metabolites_tfs_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'


# -- Predict associations
# Run Linear models
lm_f_res, n_iter = [], 20
# train, _ = list(ShuffleSplit(len(ys.ix[ion]), test_size=.2))[0]
for ion in ions:
    for train, _ in ShuffleSplit(len(ys.ix[ion]), test_size=.2, n_iter=n_iter):
        ys_train, xs_train = ys.ix[ion, train], xs.ix[tfs, train].T

        # Standardization
        xs_train /= xs_train.std()
        ys_train -= ys_train.mean()

        # Elastic Net ShuffleSplit cross-validation
        cv = ShuffleSplit(len(ys_train), test_size=.2, n_iter=n_iter)
        lm = ElasticNetCV(cv=cv).fit(xs_train, ys_train)

        # Store results
        for f, v in zip(*(tfs, lm.coef_)):
            lm_f_res.append((ion, f, v))

lm_f_res = DataFrame(lm_f_res, columns=['ion', 'feature', 'coef'])
lm_f_res['coef_abs'] = [abs(i) for i in lm_f_res['coef']]

lm_f_res['biogrid'] = [int((f, i) in interactions['tfs']['biogrid']) for i, f in lm_f_res[['ion', 'feature']].values]
lm_f_res['string'] = [int((f, i) in interactions['tfs']['string']) for i, f in lm_f_res[['ion', 'feature']].values]
lm_f_res['targets'] = [int((f, i) in interactions['tfs']['targets']) for i, f in lm_f_res[['ion', 'feature']].values]

lm_f_res['Transcription-factors'] = [acc_name[c] for c in lm_f_res['feature']]
lm_f_res['Metabolites'] = [met_name[c] for c in lm_f_res['ion']]

lm_f_res.sort('coef_abs', ascending=False).to_csv('%s/tables/metabolites_tfs_interactions.csv' % wd, index=False)
print lm_f_res.sort('coef_abs', ascending=False)


roc_table = lm_f_res.groupby(['Metabolites', 'Transcription-factors'])['coef_abs', 'targets', 'biogrid', 'string'].median().reset_index()

# Important features ROC
source_pal = {'string': '#e74c3c', 'biogrid': '#34495e', 'targets': '#2ecc71'}

sns.set(style='ticks', context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
for source in ['targets', 'biogrid', 'string']:
    curve_fpr, curve_tpr, thresholds = roc_curve(roc_table[source], roc_table['coef_abs'])
    curve_auc = auc(curve_fpr, curve_tpr)

    plt.plot(curve_fpr, curve_tpr, label='%s (area = %0.2f)' % (source, curve_auc), color=source_pal[source])

plt.plot([0, 1], [0, 1], 'k--', lw=.3)
sns.despine(trim=True)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='lower right')

plt.gcf().set_size_inches(3, 3)

plt.savefig('%s/reports/lm_dynamic_roc_tfs_gsea.pdf' % wd, bbox_inches='tight')
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
lm_res_top_features = lm_f_res[[i in order for i in lm_f_res['Metabolites']]]

t_matrix = pivot_table(lm_res_top_features, index='Metabolites', columns='Transcription-factors', values='coef', aggfunc=np.mean)
t_matrix = t_matrix.loc[:, t_matrix.std() > .01]

top_features_cor = [(feature, ion, pearsonr(xs.ix[feature], ys.ix[ion])) for feature, ion in it.product(set(lm_res_top_features['feature']), set(lm_res_top_features['ion']))]
top_features_cor = DataFrame([(f, i, c, p) for f, i, (c, p) in top_features_cor], columns=['feature', 'ion', 'cor', 'pval'])
top_features_cor['fdr'] = multipletests(top_features_cor['pval'], method='fdr_bh')[1]
signif_features = {(f, i) for f, i in top_features_cor[top_features_cor['fdr'] < .05][['feature', 'ion']].values}

cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)
sns.set(context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.clustermap(t_matrix, figsize=(4, 5), linewidth=.5, cmap=cmap, metric='correlation')

for r, c, feature, ion in lm_res_top_features[['Metabolites', 'Transcription-factors', 'feature', 'ion']].values:
    if c in g.data2d.columns and r in g.data2d.index and (feature, ion) in signif_features and t_matrix.abs().ix[r, c] > .05:
        text_x, text_y = (list(g.data2d.columns).index(c), (g.data2d.shape[0] - 1) - list(g.data2d.index).index(r))
        g.ax_heatmap.annotate('*', (text_x, text_y), xytext=(text_x + .5, text_y + .2), ha='center', va='baseline', color='#808080')

plt.savefig('%s/reports/lm_dynamic_heatmap_tfs_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'
