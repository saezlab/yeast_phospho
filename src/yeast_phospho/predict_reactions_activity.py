from __future__ import division
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from yeast_phospho.utils import pearson
from pandas import DataFrame, read_csv, melt, pivot_table
from sklearn.cross_validation import LeaveOneOut
from sklearn.linear_model import Lasso


# ---- Define filters
k_signif = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)
k_signif = list(k_signif[(k_signif.count(1) / k_signif.shape[1]) > .75].index)


# ---- Import
# Steady-state
r_activity = read_csv('%s/tables/reaction_activity_steady_state.tab' % wd, sep='\t', index_col=0)
k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0).ix[k_signif].replace(np.NaN, 0.0)
tf_activity = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0).dropna()

r_activity_g = read_csv('%s/tables/reaction_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0)
k_activity_g = read_csv('%s/tables/kinase_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0).ix[k_signif].replace(np.NaN, 0.0)
tf_activity_g = read_csv('%s/tables/tf_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0).dropna()

# Dynamic
r_activity_dyn = read_csv('%s/tables/reaction_activity_dynamic.tab' % wd, sep='\t', index_col=0)
k_activity_dyn = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0).ix[k_signif].dropna(how='all').replace(np.NaN, 0.0)
t_activity_dyn = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0).dropna()


# ---- Machine learning setup
lm = Lasso(alpha=.01, max_iter=2000)


# ---- Perform predictions
# Steady-state comparisons
steady_state = [
    (k_activity, r_activity, 'LOO', 'kinase', 'no growth'),
    (tf_activity, r_activity, 'LOO', 'tf', 'no growth'),

    (k_activity_g, r_activity_g, 'LOO', 'kinase', 'with growth'),
    (tf_activity_g, r_activity_g, 'LOO', 'tf', 'with growth')
]

# Dynamic comparisons
dynamic = [
    ((k_activity, r_activity), (k_activity_dyn, r_activity_dyn), 'dynamic', 'kinase', 'no growth'),
    ((tf_activity, r_activity), (t_activity_dyn, r_activity_dyn), 'dynamic', 'tf', 'no growth'),

    ((k_activity_g, r_activity_g), (k_activity_dyn, r_activity_dyn), 'dynamic', 'kinase', 'with growth'),
    ((tf_activity_g, r_activity_g), (t_activity_dyn, r_activity_dyn), 'dynamic', 'tf', 'with growth')
]

# Dynamic comparisons
test_comparisons = [
    ((k_activity, r_activity), (k_activity_dyn, r_activity_dyn), 'test', 'kinase', 'no growth'),
    ((tf_activity, r_activity), (t_activity_dyn, r_activity_dyn), 'test', 'tf', 'no growth'),

    ((k_activity_g, r_activity_g), (k_activity_dyn, r_activity_dyn), 'test', 'kinase', 'with growth'),
    ((tf_activity_g, r_activity_g), (t_activity_dyn, r_activity_dyn), 'test', 'tf', 'with growth')
]

lm_res, lm_betas = [], []
for xs, ys, condition, feature, growth in steady_state:
    x_features, y_features, samples = list(xs.index), list(ys.index), list(set(xs.columns).intersection(ys.columns))

    x, y = xs.ix[x_features, samples].replace(np.NaN, 0.0).T, ys.ix[y_features, samples].T

    cv = LeaveOneOut(len(samples))
    y_pred = DataFrame({samples[test]: {y_feature: lm.fit(x.ix[train], y.ix[train, y_feature]).predict(x.ix[test])[0] for y_feature in y_features} for train, test in cv})

    lm_res.extend([(condition, feature, growth, f, 'features', pearson(ys.ix[f, samples], y_pred.ix[f, samples])[0]) for f in y_features])
    lm_res.extend([(condition, feature, growth, s, 'samples', pearson(ys.ix[y_features, s], y_pred.ix[y_features, s])[0]) for s in samples])

    print '[INFO] %s, %s, %s' % (condition, feature, growth)
    print '[INFO] x_features: %d, y_features: %d, samples: %d' % (len(x_features), len(y_features), len(samples))


for (xs_train, ys_train), (xs_test, ys_test), condition, feature, growth in dynamic:
    x_features, y_features = list(set(xs_train.index).intersection(xs_test.index)), list(set(ys_train.index).intersection(ys_test.index))
    train_samples, test_samples = list(set(xs_train.columns).intersection(ys_train.columns)), list(set(xs_test.columns).intersection(ys_test.columns))

    x_train, y_train = xs_train.ix[x_features, train_samples].replace(np.NaN, 0.0).T, ys_train.ix[y_features, train_samples].T
    x_test, y_test = xs_test.ix[x_features, test_samples].replace(np.NaN, 0.0).T, ys_test.ix[y_features, test_samples].T

    y_pred = DataFrame(lm.fit(x_train, y_train).predict(x_test).T, index=y_features, columns=test_samples)

    lm_res.extend([(condition, feature, growth, f, 'features', pearson(ys_test.ix[f, test_samples], y_pred.ix[f, test_samples])[0]) for f in y_features])
    lm_res.extend([(condition, feature, growth, s, 'samples', pearson(ys_test.ix[y_features, s], y_pred.ix[y_features, s])[0]) for s in test_samples])

    betas = DataFrame(lm.coef_, index=y_features, columns=x_features)
    betas['y_features'] = betas.index
    lm_betas.extend([(condition, feature, growth, m, f, b) for m, f, b in melt(betas, id_vars='y_features').values])

    print '[INFO] %s, %s, %s' % (condition, feature, growth)
    print '[INFO] x_features: %d, y_features: %d, train_samples: %d, test_samples: %d' % (len(x_features), len(y_features), len(train_samples), len(test_samples))


for (xs_train, ys_train), (xs_test, ys_test), condition, feature, growth in test_comparisons:
    x_features, y_features = list(set(xs_train.index).intersection(xs_test.index)), list(set(ys_train.index).intersection(ys_test.index))
    train_samples, test_samples = list(set(xs_train.columns).intersection(ys_train.columns)), list(set(xs_test.columns).intersection(ys_test.columns))

    x_train, y_train = xs_train.ix[x_features, train_samples].replace(np.NaN, 0.0).T, ys_train.ix[y_features, train_samples].T
    x_test, y_test = xs_test.ix[x_features, test_samples].replace(np.NaN, 0.0).T, ys_test.ix[y_features, test_samples].T

    y_pred = {}
    for y_feature in y_features:
        outlier = y_train[y_feature].abs().argmax()
        y_pred[y_feature] = dict(zip(*(test_samples, lm.fit(x_train.drop(outlier), y_train[y_feature].drop(outlier)).predict(x_test))))

    y_pred = DataFrame(y_pred).T

    lm_res.extend([(condition, feature, growth, f, 'features', pearson(ys_test.ix[f, test_samples], y_pred.ix[f, test_samples])[0]) for f in y_features])
    lm_res.extend([(condition, feature, growth, s, 'samples', pearson(ys_test.ix[y_features, s], y_pred.ix[y_features, s])[0]) for s in test_samples])

    print '[INFO] %s, %s, %s' % (condition, feature, growth)
    print '[INFO] x_features: %d, y_features: %d, train_samples: %d, test_samples: %d' % (len(x_features), len(y_features), len(train_samples), len(test_samples))


lm_res = DataFrame(lm_res, columns=['condition', 'feature', 'growth', 'name', 'type_cor', 'cor'])
lm_res['type'] = lm_res['condition'] + '_' + lm_res['growth'] + '_' + lm_res['feature']

lm_betas = DataFrame(lm_betas, columns=['condition', 'feature', 'growth', 'ion', 'feature_name', 'beta'])
lm_betas['type'] = lm_betas['condition'] + '_' + lm_betas['growth'] + '_' + lm_betas['feature']


# ---- Plot predictions correlations
sns.set(style='ticks', palette='pastel', color_codes=True)
g = sns.FacetGrid(data=lm_res, col='condition', row='feature', legend_out=True, sharey=True, size=4, aspect=.7, ylim=(-1, 1))
g.map(plt.axhline, y=0, ls=':', c='.5')
g.map(sns.boxplot, 'type_cor', 'cor', 'growth', palette={'with growth': '#95a5a6', 'no growth': '#2ecc71'}, sym='')
g.map(sns.stripplot, 'type_cor', 'cor', 'growth', palette={'with growth': '#95a5a6', 'no growth': '#2ecc71'}, jitter=True, size=5)
g.add_legend(title='Growth rate:')
g.set_axis_labels('', 'Correlation (pearson)')
sns.despine(trim=True)
plt.savefig('%s/reports/lm_boxplot_correlations_reactions.pdf' % wd, bbox_inches='tight')
plt.close('all')
