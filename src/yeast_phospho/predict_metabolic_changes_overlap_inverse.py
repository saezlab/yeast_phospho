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


m_signif = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0)
m_signif = list(m_signif[m_signif.std(1) > .6].index)

s_info = read_csv('%s/files/strain_info.tab' % wd, sep='\t', index_col=0)
s_info = s_info[[i not in ['silent', 'low'] for i in s_info['impact']]]

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
# Dynamic comparisons
dynamic = [
    ((k_activity_dyn.copy(), metabolomics_dyn.copy()), (k_activity.copy(), metabolomics.copy()), 'dynamic', 'kinase', 'no growth'),
    ((tf_activity_dyn.copy(), metabolomics_dyn.copy()), (tf_activity.copy(), metabolomics.copy()), 'dynamic', 'tf', 'no growth'),
    ((k_tf_activity_dyn.copy(), metabolomics_dyn.copy()), (k_tf_activity.copy(), metabolomics.copy()), 'dynamic', 'overlap', 'no growth'),

    ((k_activity_dyn.copy(), metabolomics_dyn.copy()), (k_activity_g.copy(), metabolomics_g.copy()), 'dynamic', 'kinase', 'with growth'),
    ((tf_activity_dyn.copy(), metabolomics_dyn.copy()), (tf_activity_g.copy(), metabolomics_g.copy()), 'dynamic', 'tf', 'with growth'),
    ((k_tf_activity_dyn.copy(), metabolomics_dyn.copy()), (k_tf_activity_g.copy(), metabolomics_g.copy()), 'dynamic', 'overlap', 'with growth')
]

lm_res = []
for (xs_train, ys_train), (xs_test, ys_test), condition, feature, growth in dynamic:
    x_features, y_features = list(set(xs_train.index).intersection(xs_test.index)), list(set(ys_train.index).intersection(ys_test.index))
    train_samples, test_samples = list(set(xs_train.columns).intersection(ys_train.columns)), list(set(xs_test.columns).intersection(ys_test.columns))

    x_train, y_train = xs_train.ix[x_features, train_samples].replace(np.NaN, 0.0).T, ys_train.ix[y_features, train_samples].T
    x_test, y_test = xs_test.ix[x_features, test_samples].replace(np.NaN, 0.0).T, ys_test.ix[y_features, test_samples].T

    y_pred = DataFrame({y_feature: dict(zip(*(test_samples, ElasticNetCV(cv=ShuffleSplit(len(x_train), 10)).fit(x_train, y_train[y_feature]).predict(x_test)))) for y_feature in y_features}).T

    lm_res.extend([(condition, feature, growth, f, 'features', pearson(ys_test.ix[f, test_samples], y_pred.ix[f, test_samples])[0]) for f in y_features])
    lm_res.extend([(condition, feature, growth, s, 'samples', pearson(ys_test.ix[y_features, s], y_pred.ix[y_features, s])[0]) for s in test_samples])

    print '[INFO] %s, %s, %s' % (condition, feature, growth)
    print '[INFO] x_features: %d, y_features: %d, train_samples: %d, test_samples: %d' % (len(x_features), len(y_features), len(train_samples), len(test_samples))

lm_res = DataFrame(lm_res, columns=['condition', 'feature', 'growth', 'name', 'type_cor', 'cor'])
lm_res['type'] = lm_res['feature'] + '_' + lm_res['type_cor']


# ---- Plot predictions correlations
sns.set(style='ticks', palette='pastel', color_codes=True)
x_order = list(lm_res[lm_res['growth'] == 'no growth'].groupby('type').median().sort('cor', ascending=False).index)
sns.boxplot(y='type', x='cor', data=lm_res, order=x_order, hue='growth', orient='h', palette='Paired')
sns.stripplot(y='type', x='cor', data=lm_res, order=x_order, hue='growth', orient='h', size=3, jitter=True, palette='Paired')
sns.despine(trim=True)
plt.axvline(0.0, lw=.3, c='gray', alpha=0.3)
plt.xlabel('pearson correlation')
plt.ylabel('comparisons')
plt.title('Predict metabolic fold-changes')
plt.savefig(wd + 'reports/lm_boxplot_correlations_metabolites_overlap_inverse.pdf', bbox_inches='tight')
plt.close('all')
