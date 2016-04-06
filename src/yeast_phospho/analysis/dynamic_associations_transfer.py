import re
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from yeast_phospho import wd
from scipy.stats.stats import pearsonr
from sklearn.linear_model import ElasticNet
from statsmodels.stats.multitest import multipletests
from pandas import DataFrame, Series, read_csv, concat, pivot_table
from yeast_phospho.utilities import get_metabolites_name, get_proteins_name


# -- Import IDs maps
acc_name = get_proteins_name()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}

met_name = get_metabolites_name()
met_name = {'%.4f' % float(k): met_name[k] for k in met_name if len(met_name[k].split('; ')) == 1}


# -- Import direct associations
with open('%s/tables/known_associations.pickle' % wd, 'rb') as handle:
    dbs = pickle.load(handle)


# -- Import data-sets
# Nitrogen metabolism Metabolomics
metabolomics_dyn_ng = read_csv('%s/tables/metabolomics_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
# metabolomics_dyn_ng = metabolomics_dyn_ng[metabolomics_dyn_ng.std(1) > .4]
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
# metabolomics_dyn_comb = metabolomics_dyn_comb[metabolomics_dyn_comb.std(1) > .4]
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

        lm = sm.OLS(ys.ix[ion, train], xs.ix[kinases, train].T).fit_regularized(L1_wt=.5, alpha=.01)

        pred, meas = Series(lm.predict(xs.ix[kinases, test].T), index=test), ys.ix[ion, test]
        cor, pval = pearsonr(pred, meas.ix[pred.index])

        lm_res.append((ion, condition, cor, pval, lm))

lm_res = DataFrame(lm_res, columns=['ion', 'condition', 'cor', 'pval', 'lm'])
lm_res['condition'] = lm_res['condition'].replace('alpha', 'Pheromone')
lm_res['condition'] = lm_res['condition'].replace('.', 'All')
print lm_res.sort('cor')


# -- Plot
label_order = ['N_downshift', 'N_upshift', 'Rapamycin', 'NaCl', 'Pheromone']
palette = {'Rapamycin': '#D25A2B', 'N_upshift': '#5EACEC', 'N_downshift': '#4783C7', 'NaCl': '#CC2229', 'Pheromone': '#6FB353'}

#
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

#
well_predicted = set(lm_res[(lm_res['pval'] < .05) & (lm_res['cor'] > 0)]['ion'])

plot_df = lm_res[[i in well_predicted for i in lm_res['ion']]]
plot_df['name'] = [met_name[i] for i in plot_df['ion']]

top_predicted = [met_name[i] for i in well_predicted]

sns.set(style='ticks', font_scale=.75, context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.FacetGrid(plot_df, legend_out=True, aspect=1, size=2, sharex=True, sharey=False)
g.map(sns.boxplot, 'cor', 'name', sym='', orient='h', order=top_predicted, color='#CCCCCC', saturation=.1, linewidth=.3)
g.map(sns.stripplot, 'cor', 'name', 'condition', palette=palette, jitter=True, size=2, split=True, edgecolor='white', linewidth=.3, orient='h', order=top_predicted, color='#808080')
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

#
coefs = DataFrame([(ion, f, c) for ion, lm, cor in plot_df[['ion', 'lm', 'cor']].values if cor > 0 for f, c in lm.params.to_dict().items()], columns=['i', 'k', 'coef'])
coefs['kinase/phosphatase'] = [acc_name[c] for c in coefs['k']]
coefs['metabolite'] = [met_name[c] for c in coefs['i']]
coefs['reported'] = [int((k, m) in dbs['kinases']) for k, m in coefs[['k', 'i']].values]
coefs = coefs.sort(['reported', 'coef'], ascending=False)

reported = pivot_table(coefs, index='metabolite', columns='kinase/phosphatase', values='reported', aggfunc=np.max)

coefs = pivot_table(coefs, index='metabolite', columns='kinase/phosphatase', values='coef', aggfunc=np.median)
coefs = coefs.loc[:, coefs.abs().sum() > .1]

reported = reported.ix[coefs.index, coefs.columns]

cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)
sns.set(context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.clustermap(coefs, figsize=(4, 5), linewidth=.5, cmap=cmap, metric='correlation')
plt.savefig('%s/reports/linear_regression_dynamic_transfer_metabolites_heatmap_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'
