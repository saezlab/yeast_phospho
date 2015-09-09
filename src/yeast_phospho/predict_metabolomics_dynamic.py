import igraph
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from yeast_phospho.utils import pearson
from sklearn.cross_validation import LeaveOneOut
from sklearn.linear_model import Lasso
from pandas import DataFrame, read_csv, melt


m_signif = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)
m_signif = list(m_signif[m_signif.std(1) > .4].index)

k_signif = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)
k_signif = k_signif[(k_signif.count(1) / k_signif.shape[1]) > .75]

tf_signif = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0).dropna()


# ---- Import
# Steady-state
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0).ix[m_signif]
k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0).ix[k_signif].replace(np.NaN, 0.0)
tf_activity = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0).ix[tf_signif].dropna()

metabolomics_g = read_csv('%s/tables/metabolomics_steady_state_growth_rate.tab' % wd, sep='\t', index_col=0).ix[m_signif]
k_activity_g = read_csv('%s/tables/kinase_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0).ix[k_signif].replace(np.NaN, 0.0)
tf_activity_g = read_csv('%s/tables/tf_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0).ix[tf_signif].dropna()

# Dynamic
metabolomics_dyn = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0).ix[m_signif].dropna()
k_activity_dyn = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0).ix[k_signif].dropna(how='all').replace(np.NaN, 0.0)
tf_activity_dyn = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0).ix[tf_signif].dropna()


# ---- Overlap
strains = list(set(metabolomics.columns).intersection(k_activity.columns).intersection(tf_activity.columns))
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
test_comparisons = [
    ((k_activity, metabolomics), (k_activity_dyn, metabolomics_dyn), 'Test', 'kinase', 'no growth'),
    ((tf_activity, metabolomics), (tf_activity_dyn, metabolomics_dyn), 'Test', 'tf', 'no growth'),
    ((k_tf_activity, metabolomics), (k_tf_activity_dyn, metabolomics_dyn), 'Test', 'overlap', 'no growth'),

    ((k_activity_g, metabolomics_g), (k_activity_dyn, metabolomics_dyn), 'Test', 'kinase', 'with growth'),
    ((tf_activity_g, metabolomics_g), (tf_activity_dyn, metabolomics_dyn), 'Test', 'tf', 'with growth'),
    ((k_tf_activity_g, metabolomics_g), (k_tf_activity_dyn, metabolomics_dyn), 'Test', 'overlap', 'with growth')
]

# Dynamic comparisons
dynamic = [
    ((k_activity.copy(), metabolomics.copy()), (k_activity_dyn.copy(), metabolomics_dyn.copy()), 'Dynamic', 'kinase', 'no growth'),
    ((tf_activity.copy(), metabolomics.copy()), (tf_activity_dyn.copy(), metabolomics_dyn.copy()), 'Dynamic', 'tf', 'no growth'),
    ((k_tf_activity.copy(), metabolomics.copy()), (k_tf_activity_dyn.copy(), metabolomics_dyn.copy()), 'Dynamic', 'overlap', 'no growth'),

    ((k_activity_g.copy(), metabolomics_g.copy()), (k_activity_dyn.copy(), metabolomics_dyn.copy()), 'Dynamic', 'kinase', 'with growth'),
    ((tf_activity_g.copy(), metabolomics_g.copy()), (tf_activity_dyn.copy(), metabolomics_dyn.copy()), 'Dynamic', 'tf', 'with growth'),
    ((k_tf_activity_g.copy(), metabolomics_g.copy()), (k_tf_activity_dyn.copy(), metabolomics_dyn.copy()), 'Dynamic', 'overlap', 'with growth')
]


lm, lm_res = Lasso(alpha=0.01, max_iter=2000), []

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

    y_pred = DataFrame({y_feature: dict(zip(*(test_samples, lm.fit(x_train, y_train[y_feature]).predict(x_test)))) for y_feature in y_features}).T

    lm_res.extend([(condition, feature, growth, f, 'features', pearson(ys_test.ix[f, test_samples], y_pred.ix[f, test_samples])[0]) for f in y_features])
    lm_res.extend([(condition, feature, growth, s, 'samples', pearson(ys_test.ix[y_features, s], y_pred.ix[y_features, s])[0]) for s in test_samples])

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


# ---- Plot predictions correlations
sns.set(style='ticks', palette='pastel', color_codes=True, context='paper')
g = sns.FacetGrid(data=lm_res, col='condition', row='feature', legend_out=True, sharey=True, ylim=(-1, 1), col_order=['LOO', 'Dynamic', 'Test'], size=2.4, aspect=.9)
g.fig.subplots_adjust(wspace=.05, hspace=.05)
g.map(plt.axhline, y=0, ls=':', c='.5')
g.map(sns.boxplot, 'type_cor', 'cor', 'growth', palette={'with growth': '#95a5a6', 'no growth': '#2ecc71', 'dynamic': '#e74c3c'}, sym='')
g.map(sns.stripplot, 'type_cor', 'cor', 'growth', palette={'with growth': '#95a5a6', 'no growth': '#2ecc71', 'dynamic': '#e74c3c'}, jitter=True, size=5)
g.add_legend(title='Growth rate:')
g.set_axis_labels('', 'Correlation (pearson)')
sns.despine(trim=True)
plt.savefig('%s/reports/lm_metabolites_overlap.pdf' % wd, bbox_inches='tight')
plt.close('all')


# ---- Metabolites predictions correlations
m_map = read_csv('%s/files/metabolite_mz_map_kegg.txt' % wd, sep='\t')
m_map['mz'] = [float('%.2f' % i) for i in m_map['mz']]
m_map = m_map.drop_duplicates('mz').drop_duplicates('formula')
m_map = m_map.groupby('mz')['name'].apply(lambda i: '; '.join(i)).to_dict()

pred_met = lm_res[(lm_res['condition'] == 'LOO') & (lm_res['growth'] == 'dynamic') & (lm_res['type_cor'] == 'features') & (lm_res['feature'] != 'overlap')]
pred_met['metabolite'] = [m_map[i] for i in pred_met['name']]
pred_met = pred_met.sort('cor', ascending=False)

acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)['gene'].to_dict()

order = list(pred_met[pred_met['feature'] == 'tf'].sort('cor', ascending=False)['metabolite'])
sns.set(style='ticks', palette='pastel', color_codes=True, context='paper')
g = sns.FacetGrid(data=pred_met, legend_out=True, sharey=False, xlim=(-.3, 1), size=4, aspect=1)
g.fig.subplots_adjust(wspace=.05, hspace=.05)
g.map(plt.axvline, x=0, ls=':', c='.5')
g.map(sns.barplot, 'cor', 'metabolite', 'feature', palette={'tf': '#34495e', 'kinase': '#3498db'}, ci=None, orient='h', lw=0, order=order)
g.add_legend(title='Feature type:')
g.set_axis_labels('Correlation (pearson)', 'Metabolite')
sns.despine(trim=True)
plt.savefig('%s/reports/lm_metabolites_overlap_list.pdf' % wd, bbox_inches='tight')
plt.close('all')


# ---- Dynamic associations
dynamic_associations = [
    (k_activity_dyn, metabolomics_dyn, 'kinase'),
    (tf_activity_dyn, metabolomics_dyn, 'tf'),
    (k_tf_activity_dyn, metabolomics_dyn, 'overlap')
]

lm_dyn_betas = DataFrame()
for xs_test, ys_test, feature_type in dynamic_associations:
    samples, x_features, y_features = list(set(xs_test).intersection(ys_test)), list(xs_test.index), list(ys_test.index)

    x_train, y_train = xs_test[samples].T, ys_test[samples].T

    lm_betas = DataFrame(lm.fit(x_train, y_train).coef_, index=y_features, columns=x_features)

    sns.set(style='white', palette='pastel', color_codes=True, context='paper')
    xticklabels = [acc_name[i].split(';')[0] for i in lm_betas.columns]
    yticklabels = [m_map[i] for i in lm_betas.index]
    sns.clustermap(lm_betas, figsize=(int(lm_betas.shape[1] * .3), 5), xticklabels=xticklabels, yticklabels=yticklabels)
    plt.savefig('%s/reports/lm_metabolites_overlap_betas_%s.pdf' % (wd, feature_type), bbox_inches='tight')
    plt.close('all')

    lm_betas['metabolite'] = lm_betas.index
    lm_betas = melt(lm_betas, id_vars='metabolite', value_name='beta', var_name='feature')
    lm_betas = lm_betas[lm_betas['beta'] != 0]
    lm_betas['type'] = feature_type

    lm_dyn_betas = lm_dyn_betas.append(lm_betas, ignore_index=True)

    print '[INFO] %s' % feature_type


plot_df = []
for m in set(pred_met.head()['name']):
    best_k = lm_dyn_betas.ix[lm_dyn_betas[(lm_dyn_betas['metabolite'] == m) & (lm_dyn_betas['type'] == 'kinase')]['beta'].argmax(), 'feature']
    best_t = lm_dyn_betas.ix[lm_dyn_betas[(lm_dyn_betas['metabolite'] == m) & (lm_dyn_betas['type'] == 'tf')]['beta'].argmax(), 'feature']

    plot_df.extend([(x, y, c, m, best_k, 'kinase') for c in conditions for x, y in zip(*(metabolomics_dyn.ix[m, conditions], k_activity_dyn.ix[best_k, conditions]))])
    plot_df.extend([(x, y, c, m, best_t, 'tf') for c in conditions for x, y in zip(*(metabolomics_dyn.ix[m, conditions], tf_activity_dyn.ix[best_t, conditions]))])

    print '[INFO] %s' % m

plot_df = DataFrame(plot_df, columns=['x', 'y', 'condition', 'metabolite', 'feature', 'feature_type'])
plot_df['metabolite'] = [m_map[m] for m in plot_df['metabolite']]


sns.set(style='ticks', palette='pastel', color_codes=True, context='paper')
g = sns.lmplot(x='x', y='y', data=plot_df, row='metabolite', hue='feature_type', sharex=False, sharey=False, size=2.5, aspect=1, scatter_kws={'s': 50, 'alpha': .6}, palette={'tf': '#34495e', 'kinase': '#3498db'})
g.despine(trim=True)
g.set_xlabels('Metabolite (fold-change)')
g.set_ylabels('Feature (activity)')
plt.savefig('%s/reports/lm_metabolites_overlap_list_scatter.pdf' % wd, bbox_inches='tight')
plt.close('all')


# ---- Plot network
beta_thres = .5

network = lm_dyn_betas[(lm_dyn_betas['beta'].abs() > beta_thres) & (lm_dyn_betas['type'] != 'overlap')]
network['metabolite'] = [str(m) for m in network['metabolite']]

network_i = igraph.Graph(directed=False)
network_i.add_vertices(list(set(network['metabolite']).union(network['feature'])))
network_i.add_edges([(m, p) for m, p in network[['metabolite', 'feature']].values])
print '[INFO] Network: ', network_i.summary()


# Set nodes attributes
def node_name(i):
    if i in kinases:
        return acc_name[i].split(';')[0]
    elif i in tfs:
        return acc_name[i].split(';')[0]
    else:
        return m_map[float(i)]
network_i.vs['label'] = [node_name(v) for v in network_i.vs['name']]


def node_shape(i):
    if i in kinases:
        return 'square'
    elif i in tfs:
        return 'square'
    else:
        return 'circle'
network_i.vs['shape'] = [node_shape(v) for v in network_i.vs['name']]


def node_colour(i):
    if i in kinases:
        return '#3498db'
    elif i in tfs:
        return '#34495e'
    else:
        return '#e74c3c'
network_i.vs['color'] = [node_colour(v) for v in network_i.vs['name']]


def node_label_color(i):
    if i in kinases:
        return 'white'
    elif i in tfs:
        return 'white'
    else:
        return 'black'
network_i.vs['label_color'] = [node_label_color(v) for v in network_i.vs['name']]


# Set edges attributes
network_i.es['color'] = ['#e74c3c' if e < 0 else '#2ecc71' for e in network['beta']]

# Calculate layout
layout = network_i.layout_fruchterman_reingold(maxiter=10000, area=50 * (len(network_i.vs) ** 2))
print '[INFO] Network layout created: ', network_i.summary()

# Export network
igraph.plot(
    network_i,
    layout=layout,
    bbox=(0, 0, 360, 360),
    vertex_label_size=5,
    vertex_frame_width=0,
    vertex_size=20,
    edge_width=1.,
    target='%s/reports/lm_metabolites_overlap_network.pdf' % wd
)
print '[INFO] Network exported: ', network_i.summary()
