import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection.univariate_selection import GenericUnivariateSelect
from sklearn.pipeline import Pipeline
from statsmodels.stats.multitest import multipletests
from yeast_phospho import wd
from yeast_phospho.utils import spearman
from pandas import DataFrame, read_csv, Index
from sklearn.cross_validation import LeaveOneOut
from sklearn.linear_model import LinearRegression

sns.set_style('ticks')

# Import acc map to name form uniprot
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)['gene'].to_dict()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}

# Import metabolites map
m_map = read_csv('%s/files/metabolomics/metabolite_mz_map_kegg.txt' % wd, sep='\t')
m_map['mz'] = Index([str(i) for i in m_map['mz']], dtype=str)
m_map = m_map.groupby('mz')['name'].apply(lambda x: '; '.join(x)).to_dict()

# Import kinase activity
k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)

# Import metabolomics
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)
metabolomics.index = Index([str(i) for i in metabolomics.index], dtype=str)

# Overlapping kinases/phosphatases knockout
strains = list(set(k_activity.columns).intersection(set(metabolomics.columns)))
k_activity, metabolomics = k_activity[strains], metabolomics[strains]

# ---- Steady-state: predict metabolites FC with kinases
x, y = k_activity[strains].replace(np.NaN, 0.0).T, metabolomics[strains].T
m_predicted = DataFrame({strains[test]: dict(zip(*(y.columns, LinearRegression().fit(x.ix[train], y.ix[train]).predict(x.ix[test])[0]))) for train, test in LeaveOneOut(len(strains))})

# Plot predicted prediction scores
m_score = [(m, spearman(metabolomics.ix[m, strains], m_predicted.ix[m, strains])) for m in m_predicted.index]
m_score = DataFrame([(m, c, p) for m, (c, p) in m_score], columns=['metabolite', 'correlation', 'pvalue'])
m_score = m_score.set_index('metabolite')
m_score['adjpvalue'] = multipletests(m_score['pvalue'], method='fdr_bh')[1]
m_score['signif'] = ['FDR < 5%' if x < 0.05 else ' FDR >= 5%' for x in m_score['adjpvalue']]
m_score['name'] = [m_map[m] if m in m_map else m for m in m_score.index]
m_score = m_score.sort('correlation', ascending=False)
print 'Mean correlation metabolites: ', m_score['correlation'].mean()

s_score = [(s, spearman(metabolomics.ix[metabolomics.index, s], m_predicted.ix[metabolomics.index, s])) for s in strains]
s_score = DataFrame([(m, c, p) for m, (c, p) in s_score], columns=['strain', 'correlation', 'pvalue'])
s_score = s_score.set_index('strain')
s_score['adjpvalue'] = multipletests(s_score['pvalue'], method='fdr_bh')[1]
s_score['signif'] = ['FDR < 5%' if x < 0.05 else ' FDR >= 5%' for x in s_score['adjpvalue']]
s_score['name'] = [acc_name[s] for s in s_score.index]
s_score = s_score.sort('correlation', ascending=False)
print 'Mean correlation samples: ', s_score['correlation'].mean()

plot_df = m_score[m_score['adjpvalue'] < 0.1].index
plot_df = DataFrame([(m_score.ix[m, 'name'], m_predicted.ix[m, s], metabolomics.ix[m, s], m_score.ix[m, 'signif']) for m in plot_df for s in strains])
plot_df.columns = ['metabolite', 'predicted', 'measured', 'signif']

colour_pallete = list(reversed(sns.color_palette('Paired')[:2]))
g = sns.lmplot(x='measured', y='predicted', col='metabolite', hue='signif', data=plot_df, col_wrap=12, palette=colour_pallete, sharex=False, sharey=False, scatter_kws={'s': 80})
plt.savefig('%s/reports/lm_metabolites_steadystate_corr.png' % wd, bbox_inches='tight')
g.set_axis_labels('Measured', 'Predicted').fig.subplots_adjust(wspace=.02)
plt.close('all')
print '[INFO] Plot done!'

plot_df = s_score[s_score['adjpvalue'] < 0.1].index
plot_df = DataFrame([(acc_name[s], m_predicted.ix[m, s], metabolomics.ix[m, s], s_score.ix[s, 'signif']) for s in plot_df for m in m_predicted.index])
plot_df.columns = ['strain', 'predicted', 'measured', 'signif']

colour_pallete = list(reversed(sns.color_palette('Paired')[:2]))
g = sns.lmplot(x='measured', y='predicted', col='strain', hue='signif', data=plot_df, col_wrap=12, palette=colour_pallete, sharex=False, sharey=False, scatter_kws={'s': 80})
plt.savefig('%s/reports/lm_samples_steadystate_corr.png' % wd, bbox_inches='tight')
g.set_axis_labels('Measured', 'Predicted').fig.subplots_adjust(wspace=.02)
plt.close('all')
print '[INFO] Plot done!'

# # Correlation between steady-state and dynamic
# kinases_ov = set(kinase_df.index).intersection(dyn_kinase_df.index)
# metabol_ov = set(metabol_df.index).intersection(dyn_metabol_df.index)

# # Import dynamic data-sets
# dyn_kinase_df = read_csv(wd + 'tables/kinase_enrichment_dynamic_df.tab', sep='\t', index_col=0)
#
# dyn_metabol_df = read_csv(wd + 'tables/dynamic_metabolomics.tab', sep='\t', index_col=0)
# dyn_metabol_df.index = Index(dyn_metabol_df.index, dtype=str)
# # Prediction of the dynamic data-set
# # Import metabolites map
# m_map = read_csv(wd + 'files/metabolomics/metabolite_mz_map_kegg.txt', sep='\t')  # _adducts
# m_map['mz'] = ['%.2f' % c for c in m_map['mz']]
# dyn_phospho_fc = read_csv(wd + 'tables/dynamic_phosphoproteomics.tab', sep='\t', index_col='site')
#
# X, X_test = kinase_df[kinase_df.count(1) > (115 * 0.95)].replace(np.NaN, 0.0).T, dyn_kinase_df[dyn_kinase_df.count(1) > (18 * 0.95)].replace(np.NaN, 0.0).T
# Y, Y_test = metabol_df.dropna().T, dyn_metabol_df.dropna().T
#
# # Sort and re-shape data-sets
# dyncond, kinases, metabol = Y_test.index.values, list(set(X.columns).intersection(X_test.columns)), list(set(Y.columns).intersection(Y_test.columns))
#
# X, Y = X.ix[strains, kinases], Y.ix[strains, metabol]
# X_test, Y_test = X_test.ix[dyncond, kinases], Y_test.ix[dyncond, metabol]
# print '[INFO] %d kinases, %d metabolites' % (len(kinases), len(metabol))
#
# # Run linear models
# models = {m: LinearRegression().fit(X, Y[m]) for m in metabol}
#
# # Run predictions
# Y_test_predict = DataFrame({m: models[m].predict(X_test) for m in metabol}, index=dyncond)
#
# # Predict conditions across metabolites
# cor_pred_c = [(c, pearsonr(Y_test.ix[c, metabol], Y_test_predict.ix[c, metabol]), Y_test.ix[c, metabol].values, Y_test_predict.ix[c, metabol].values) for c in dyncond]
# cor_pred_c = DataFrame([(m, c, p, x[i], y[i]) for m, (c, p), x, y in cor_pred_c for i in range(len(x))], columns=['cond', 'cor', 'pvalue', 'y_true', 'y_pred'])
#
# titles = {k: '%s (r=%.2f)' % (k, c) for k, c in cor_pred_c.groupby('cond').first()['cor'].to_dict().items()}
# g = sns.lmplot('y_true', 'y_pred', cor_pred_c, col='cond', col_wrap=6, size=3, scatter_kws={'s': 50, 'alpha': .8}, line_kws={'c': '#FFFFFF', 'alpha': .7}, ci=70, palette='muted', sharex=False, sharey=False, col_order=dyncond)
# [ax.set_title(titles[title]) for ax, title in zip(g.axes.flat, dyn_metabol_df.columns)]
# plt.savefig(wd + 'reports/%s_lm_pred_conditions.pdf' % version, bbox_inches='tight')
# plt.close('all')
#
# # Predict metabolites across conditions
# cor_pred_m = [(m, pearsonr(Y_test.ix[dyncond, m], Y_test_predict.ix[dyncond, m]), Y_test.ix[dyncond, m].values, Y_test_predict.ix[dyncond, m].values) for m in metabol]
# cor_pred_m = DataFrame([(m, c, p, x[i], y[i]) for m, (c, p), x, y in cor_pred_m for i in range(len(x))], columns=['met', 'cor', 'pvalue', 'y_true', 'y_pred'])
#
# col_order = cor_pred_m.groupby('met').first().sort('cor', ascending=False).index
# titles = {k: '%s (r=%.2f)' % (k, c) for k, c in cor_pred_m.groupby('met').first()['cor'].to_dict().items()}
# g = sns.lmplot('y_true', 'y_pred', cor_pred_m, col='met', col_wrap=9, size=3, scatter_kws={'s': 50, 'alpha': .8}, line_kws={'c': '#FFFFFF', 'alpha': .7}, ci=70, palette='muted', sharex=False, sharey=False, col_order=col_order)
# [ax.set_title(titles[title]) for ax, title in zip(g.axes.flat, col_order)]
# plt.savefig(wd + 'reports/%s_lm_pred_metabolites.pdf' % version, bbox_inches='tight')
# plt.close('all')
#
# print '[INFO] Predictions done'
