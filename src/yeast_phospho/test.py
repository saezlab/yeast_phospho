import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, Series, read_csv


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

lm = Lasso(alpha=.01).fit(tf_activity[strains].T, metabolomics.ix[m, strains].T)
features = Series(lm.coef_, index=tf_activity.index).abs().sort(inplace=False)

sns.set(style='ticks', palette='pastel', color_codes=True)

plot_df = [(x, y, k, m, 'dynamic') for k in features.tail(3).index for x, y in zip(metabolomics_dyn.ix[m, conditions], tf_activity_dyn.ix[k, conditions])]
plot_df.extend([(x, y, k, m, 'steady-state') for k in features.tail(3).index for x, y in zip(metabolomics.ix[m, strains], tf_activity.ix[k, strains])])
plot_df = DataFrame(plot_df, columns=['x', 'y', 'kinase', 'metabolite', 'data-set'])

g = sns.lmplot(x='x', y='y', data=plot_df, col='kinase', row='data-set', sharex=False, sharey=False)
g.set_axis_labels('fold-change (ion %.2f)' % m, 'kinase activity')
plt.savefig(wd + 'reports/lm_test_case.pdf', bbox_inches='tight')
plt.close('all')
