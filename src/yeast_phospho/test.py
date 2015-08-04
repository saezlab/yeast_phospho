import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.stats.misc import zscore
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.regression import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics.scorer import mean_squared_error_scorer
from sklearn.svm import SVR, LinearSVR
from yeast_phospho import wd
from yeast_phospho.utils import pearson
from pandas import DataFrame, read_csv, Series
from sklearn.cross_validation import LeaveOneOut, KFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet, ElasticNetCV, RidgeCV, LassoCV, LinearRegression


m_signif = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)
m_signif = m_signif[m_signif.std(1) > .4]
m_signif = list(m_signif[(m_signif.abs() > .8).sum(1) > 0].index)

s_info = read_csv('%s/files/strain_info.tab' % wd, sep='\t', index_col=0)
# s_info = s_info[[i not in ['silent'] for i in s_info['impact']]]

# ---- Import
# Steady-state
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0).ix[m_signif]
k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0).dropna()
tf_activity = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0).dropna()

metabolomics_g = read_csv('%s/tables/metabolomics_steady_state_growth_rate.tab' % wd, sep='\t', index_col=0).ix[m_signif]
k_activity_g = read_csv('%s/tables/kinase_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0).dropna()
tf_activity_g = read_csv('%s/tables/tf_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0).dropna()

# Dynamic
metabolomics_dyn = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0).ix[m_signif].dropna()
k_activity_dyn = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0).dropna()
tf_activity_dyn = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0).dropna()


# ---- Overlap
strains = list(set(metabolomics.columns).intersection(k_activity.columns).intersection(tf_activity.columns).intersection(s_info.index))
conditions = list(set(metabolomics_dyn.columns).intersection(k_activity_dyn.columns).intersection(tf_activity_dyn.columns))
metabolites = list(set(metabolomics.index).intersection(metabolomics_dyn.index))
kinases = list(set(k_activity.index).intersection(k_activity_dyn.index))
tfs = list(set(tf_activity.index).intersection(tf_activity_dyn.index))

metabolomics, k_activity, tf_activity = metabolomics.ix[metabolites, strains], k_activity.ix[kinases, strains], tf_activity.ix[tfs, strains]
metabolomics_g, k_activity_g, tf_activity_g = metabolomics_g.ix[metabolites, strains], k_activity_g.ix[kinases, strains], tf_activity_g.ix[tfs, strains]
metabolomics_dyn, k_activity_dyn, tf_activity_dyn = metabolomics_dyn.ix[metabolites, conditions], k_activity_dyn.ix[kinases, conditions], tf_activity_dyn.ix[tfs, conditions]

k_tf_activity = k_activity.append(tf_activity)
k_tf_activity_g = k_activity_g.append(tf_activity_g)
k_tf_activity_dyn = k_activity_dyn.append(tf_activity_dyn)

# ---- Perform predictions
# Steady-state comparisons
steady_state = [
    (k_activity.copy(), metabolomics.copy(), 'LOO', 'kinase', 'no growth'),
    (tf_activity.copy(), metabolomics.copy(), 'LOO', 'tf', 'no growth'),
    (k_tf_activity.copy(), metabolomics.copy(), 'LOO', 'overlap', 'no growth'),

    (k_activity_g.copy(), metabolomics_g.copy(), 'LOO', 'kinase', 'with growth'),
    (tf_activity_g.copy(), metabolomics_g.copy(), 'LOO', 'tf', 'with growth'),
    (k_tf_activity_g.copy(), metabolomics_g.copy(), 'LOO', 'overlap', 'with growth'),

    (k_activity_dyn.copy(), metabolomics_dyn.copy(), 'LOO', 'kinase', 'dynamic'),
    (tf_activity_dyn.copy(), metabolomics_dyn.copy(), 'LOO', 'tf', 'dynamic'),
    (k_tf_activity_dyn.copy(), metabolomics_dyn.copy(), 'LOO', 'overlap', 'dynamic')
]

# Dynamic comparisons
dynamic = [
    ((k_activity.copy(), metabolomics.copy()), (k_activity_dyn.copy(), metabolomics_dyn.copy()), 'dynamic', 'kinase', 'no growth'),
    ((tf_activity.copy(), metabolomics.copy()), (tf_activity_dyn.copy(), metabolomics_dyn.copy()), 'dynamic', 'tf', 'no growth'),
    ((k_tf_activity.copy(), metabolomics.copy()), (k_tf_activity_dyn.copy(), metabolomics_dyn.copy()), 'dynamic', 'overlap', 'no growth'),

    ((k_activity_g.copy(), metabolomics_g.copy()), (k_activity_dyn.copy(), metabolomics_dyn.copy()), 'dynamic', 'kinase', 'with growth'),
    ((tf_activity_g.copy(), metabolomics_g.copy()), (tf_activity_dyn.copy(), metabolomics_dyn.copy()), 'dynamic', 'tf', 'with growth'),
    ((k_tf_activity_g.copy(), metabolomics_g.copy()), (k_tf_activity_dyn.copy(), metabolomics_dyn.copy()), 'dynamic', 'overlap', 'with growth')
]

m = 604.07

x, y = tf_activity[strains].T, metabolomics.ix[m, strains].T
lm = Lasso(alpha=.01).fit(x, y)
features = Series(lm.coef_, index=tf_activity.index).abs().sort(inplace=False)

sns.set(style='ticks', palette='pastel', color_codes=True)

plot_df = [(x, y, k, m, 'dynamic') for k in features.tail(3).index for x, y in zip(metabolomics_dyn.ix[m, conditions], tf_activity_dyn.ix[k, conditions])]
plot_df.extend([(x, y, k, m, 'steady-state') for k in features.tail(3).index for x, y in zip(metabolomics.ix[m, strains], tf_activity.ix[k, strains])])
plot_df = DataFrame(plot_df, columns=['x', 'y', 'kinase', 'metabolite', 'data-set'])

g = sns.lmplot(x='x', y='y', data=plot_df, col='kinase', row='data-set', sharex=False, sharey=False)
g.set_axis_labels('fold-change (ion %.2f)' % m, 'kinase activity')
plt.savefig(wd + 'reports/test_case/lm_test_case.pdf', bbox_inches='tight')
plt.close('all')


m = 604.07
x, y = tf_activity.loc[tfs, strains].T, metabolomics.ix[m, strains].T
tf_betas = DataFrame({strains[test]: dict(zip(*(tfs, Lasso(alpha=.01).fit(x.ix[train], y.ix[train]).coef_))) for train, test in LeaveOneOut(len(x))})
tf_betas = tf_betas.ix[tf_betas.std(1).sort(inplace=False, ascending=False).index]

f, ax = plt.subplots(figsize=(20, 6))
g = sns.boxplot(tf_betas.T, ax=ax)
g.set_xticklabels(tf_betas.index, rotation=70)
sns.despine(trim=True, ax=ax)
ax.set_ylabel('TF beta')
plt.savefig(wd + 'reports/test_case/lm_test_case_betas_boxplot.pdf', bbox_inches='tight')
plt.close('all')


x, y = tf_activity.loc[tfs, strains].T, metabolomics[strains].T
res = [(t, p, s, strains[test], m) for train, test in LeaveOneOut(len(x)) for m in metabolites for s, t, p in zip(strains, y.ix[strains], Lasso(alpha=.01).fit(x.ix[train], y.ix[train, m]).predict(x.ix[strains]))]
res = DataFrame(res, columns=['y_true', 'y_pred', 'strain', 'strain_loo', 'met'])
res['type'] = ['predicted' if s1 == s2 else 'trained' for s1, s2 in res[['strain', 'strain_loo']].values]

sns.set(style='ticks', palette='pastel', color_codes=True)
sns.lmplot(x='y_true', y='y_pred', data=res, col='type', row='met', hue='type', palette='Set1', ci=None, sharey=False, sharex=True, scatter_kws={'s': 80, 'alpha': 1})
plt.savefig(wd + 'reports/test_case/lm_test_case_betas_scatter.pdf', bbox_inches='tight')
plt.close('all')
print 'Done'


x, y = tf_activity.loc[tfs, strains].T, metabolomics[strains].T

lm, cv = Lasso(alpha=.01), LeaveOneOut(len(x))

res_predicted = [(y.ix[test, m].values[0], lm.fit(x.ix[train], y.ix[train, m]).predict(x.ix[test])[0], strains[test], m, 'predicted') for train, test in cv for m in metabolites]
res_trained = [(y.ix[test, m].values[0], lm.fit(x, y[m]).predict(x.ix[test])[0], strains[test], m, 'trained') for train, test in cv for m in metabolites]
res_all = [(t, p, s, m, 'all') for train, test in cv for m in metabolites for t, p, s in zip(*(y[m], lm.fit(x.ix[train], y.ix[train, m]).predict(x), strains))]

res_predicted.extend(res_trained)
res_predicted.extend(res_all)

res = DataFrame(res_predicted, columns=['y_true', 'y_pred', 'strain', 'met', 'type'])

sns.set(style='ticks', palette='pastel', color_codes=True)
g = sns.lmplot(x='y_true', y='y_pred', data=res, col='type', row='met', hue='type', palette='Set1', ci=None, sharey=False, sharex=False, scatter_kws={'s': 80, 'alpha': 1})
plt.savefig(wd + 'reports/test_case/lm_test_case_betas_scatter.pdf', bbox_inches='tight')
plt.close('all')
print 'Done'


