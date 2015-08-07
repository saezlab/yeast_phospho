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
m_signif = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)
m_signif = m_signif[m_signif.std(1) > .4]
m_signif = list(m_signif[(m_signif.abs() > .8).sum(1) > 0].index)

k_signif = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)
k_signif = list(k_signif[(k_signif.count(1) / k_signif.shape[1]) > .75].index)


# ---- Import
# Steady-state
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0).ix[m_signif]
k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0).ix[k_signif].replace(np.NaN, 0.0)
tf_activity = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0).dropna()

metabolomics_g = read_csv('%s/tables/metabolomics_steady_state_growth_rate.tab' % wd, sep='\t', index_col=0).ix[m_signif]
k_activity_g = read_csv('%s/tables/kinase_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0).ix[k_signif].replace(np.NaN, 0.0)
tf_activity_g = read_csv('%s/tables/tf_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0).dropna()

# Dynamic
metabolomics_dyn = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0)
k_activity_dyn = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0).ix[k_signif].dropna(how='all').replace(np.NaN, 0.0)
t_activity_dyn = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0).dropna()


# ---- Machine learning setup
lm = Lasso(alpha=.01, max_iter=2000)


# ---- Perform predictions
# Steady-state comparisons
steady_state = [
    (k_activity, metabolomics, 'LOO', 'kinase', 'no growth'),
    (tf_activity, metabolomics, 'LOO', 'tf', 'no growth'),

    (k_activity_g, metabolomics_g, 'LOO', 'kinase', 'with growth'),
    (tf_activity_g, metabolomics_g, 'LOO', 'tf', 'with growth')
]

# Dynamic comparisons
dynamic = [
    ((k_activity, metabolomics), (k_activity_dyn, metabolomics_dyn), 'dynamic', 'kinase', 'no growth'),
    ((tf_activity, metabolomics), (t_activity_dyn, metabolomics_dyn), 'dynamic', 'tf', 'no growth'),

    ((k_activity_g, metabolomics_g), (k_activity_dyn, metabolomics_dyn), 'dynamic', 'kinase', 'with growth'),
    ((tf_activity_g, metabolomics_g), (t_activity_dyn, metabolomics_dyn), 'dynamic', 'tf', 'with growth')
]

# Dynamic comparisons
test_comparisons = [
    ((k_activity, metabolomics), (k_activity_dyn, metabolomics_dyn), 'test', 'kinase', 'no growth'),
    ((tf_activity, metabolomics), (t_activity_dyn, metabolomics_dyn), 'test', 'tf', 'no growth'),

    ((k_activity_g, metabolomics_g), (k_activity_dyn, metabolomics_dyn), 'test', 'kinase', 'with growth'),
    ((tf_activity_g, metabolomics_g), (t_activity_dyn, metabolomics_dyn), 'test', 'tf', 'with growth')
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
plt.savefig('%s/reports/lm_boxplot_correlations_metabolites.pdf' % wd, bbox_inches='tight')
plt.close('all')


# ---- Import metabolites map
m_map = read_csv('%s/files/metabolite_mz_map_kegg.txt' % wd, sep='\t')
m_map['mz'] = [float('%.2f' % i) for i in m_map['mz']]
m_map = m_map.drop_duplicates('mz').drop_duplicates('formula')
m_map = m_map.groupby('mz')['name'].apply(lambda i: '; '.join(i)).to_dict()


# ---- Import YORF names
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)['gene'].to_dict()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}


# ---- Feature selection
conditions = ['_'.join(['dynamic', 'no growth', 'tf']), '_'.join(['dynamic', 'no growth', 'kinase']), '_'.join(['dynamic', 'all', 'tf']), '_'.join(['dynamic', 'all', 'kinase'])]

for condition in conditions:
    plot_df = pivot_table(lm_betas[lm_betas['type'] == condition], values='beta', columns='feature_name', index='ion', aggfunc=np.median)
    plot_df = plot_df.loc[:, (plot_df != 0).sum() != 0]

    ions_cor = lm_res[np.bitwise_and(lm_res['type'] == condition, lm_res['type_cor'] == 'features')].sort('cor', ascending=False).set_index('name')['cor']
    plot_df = plot_df.ix[list(ions_cor.index)]

    plot_df = plot_df[[i in m_map for i in plot_df.index]]

    plot_df.index = ['%s (cor: %.2f)' % (m_map[i], ions_cor.ix[i]) for i in plot_df.index]
    plot_df.columns = [acc_name[i] for i in plot_df.columns]

    plot_df.to_csv('%s/tables/lm_feature_selection_%s.txt' % (wd, condition), sep='\t')

    sns.set(style='ticks', palette='pastel', color_codes=True)
    sns.clustermap(plot_df, row_cluster=True, annot=True, fmt='.1f', figsize=(25, int(len(plot_df) * .45)), annot_kws={'size': 8})
    plt.savefig('%s/reports/lm_feature_selection_%s.pdf' % (wd, condition), bbox_inches='tight')
    plt.close('all')

    print '[INFO] %s features plotted' % condition
