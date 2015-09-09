import igraph
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from yeast_phospho.utils import pearson
from pandas.stats.misc import zscore
from matplotlib.gridspec import GridSpec
from sklearn.cross_validation import LeaveOneOut
from sklearn.linear_model import Lasso
from pandas import DataFrame, read_csv, melt, pivot_table

dyn_xorder = [
    'N_downshift_5min', 'N_downshift_9min', 'N_downshift_15min', 'N_downshift_25min', 'N_downshift_44min', 'N_downshift_79min',
    'N_upshift_5min', 'N_upshift_9min', 'N_upshift_15min', 'N_upshift_25min', 'N_upshift_44min', 'N_upshift_79min',
    'Rapamycin_5min', 'Rapamycin_9min', 'Rapamycin_15min', 'Rapamycin_25min', 'Rapamycin_44min', 'Rapamycin_79min'
]


# ---- Import IDs maps
m_map = read_csv('%s/files/metabolite_mz_map_kegg.txt' % wd, sep='\t')
m_map['mz'] = [float('%.2f' % i) for i in m_map['mz']]
m_map = m_map.drop_duplicates('mz').drop_duplicates('formula')
m_map = m_map.groupby('mz')['name'].apply(lambda i: '; '.join(i)).to_dict()

acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)['gene'].to_dict()


# ---- Import
# Dynamic
metabolomics_dyn = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0).dropna()
metabolomics_dyn = metabolomics_dyn[metabolomics_dyn.std(1) > .4]

k_activity_dyn = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0)
k_activity_dyn = k_activity_dyn[(k_activity_dyn.count(1) / k_activity_dyn.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_dyn = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0).replace(np.NaN, 0.0)


# ---- Perform predictions
# Steady-state comparisons
steady_state = [
    (k_activity_dyn.copy(), metabolomics_dyn.copy(), 'kinase'),
    (tf_activity_dyn.copy(), metabolomics_dyn.copy(), 'tf'),
]

lm = Lasso(alpha=0.01, max_iter=2000)
lm_res, lm_betas = [], []

for xs, ys, feature in steady_state:
    x_features, y_features, samples = list(xs.index), list(ys.index), list(set(xs.columns).intersection(ys.columns))
    x, y = xs.ix[x_features, samples].T, ys.ix[y_features, samples].T

    y_pred = {}
    for train, test in LeaveOneOut(len(samples)):
        y_pred[samples[test]] = {}

        for y_feature in y_features:
            model = lm.fit(x.ix[train], zscore(y.ix[train, y_feature]))

            y_pred[samples[test]][y_feature] = model.predict(x.ix[test])[0]

            lm_betas.extend([(feature, y_feature, k, v) for k, v in dict(zip(*(x.columns, model.coef_))).items()])

    y_pred = DataFrame(y_pred)

    lm_res.extend([(feature, f, 'features', pearson(ys.ix[f, samples], y_pred.ix[f, samples])[0]) for f in y_features])
    lm_res.extend([(feature, s, 'samples', pearson(ys.ix[y_features, s], y_pred.ix[y_features, s])[0]) for s in samples])

    print '[INFO] %s' % feature
    print '[INFO] x_features: %d, y_features: %d, samples: %d' % (len(x_features), len(y_features), len(samples))


lm_res = DataFrame(lm_res, columns=['feature', 'name', 'type_cor', 'cor'])

lm_betas = DataFrame(lm_betas, columns=['type', 'metabolite', 'feature', 'beta'])
lm_betas_kinase = pivot_table(lm_betas[lm_betas['type'] == 'kinase'], values='beta', index='feature', columns='metabolite', aggfunc=np.median)
lm_betas_tf = pivot_table(lm_betas[lm_betas['type'] == 'tf'], values='beta', index='feature', columns='metabolite', aggfunc=np.median)

pred_met = lm_res[[i in m_map for i in lm_res['name']]]
pred_met['metabolite'] = [m_map[i] for i in pred_met['name']]
pred_met = pred_met.dropna()
pred_met = pred_met.sort('cor', ascending=False)

pred_met_bf = []
for m in pred_met.ix[pred_met['feature'] == 'kinase', 'name'].head():
    best_k = lm_betas_kinase[m].abs().argmax()
    best_t = lm_betas_tf[m].abs().argmax()

    pred_met_bf.extend([(x, y, c, m, best_k, 'kinase') for c in dyn_xorder for x, y in zip(*(metabolomics_dyn.ix[m, dyn_xorder], k_activity_dyn.ix[best_k, dyn_xorder]))])
    pred_met_bf.extend([(x, y, c, m, best_t, 'tf') for c in dyn_xorder for x, y in zip(*(metabolomics_dyn.ix[m, dyn_xorder], tf_activity_dyn.ix[best_t, dyn_xorder]))])

    print '[INFO] %s' % m

pred_met_bf = DataFrame(pred_met_bf, columns=['x', 'y', 'condition', 'metabolite', 'feature', 'feature_type'])
pred_met_bf['metabolite'] = [m_map[m] for m in pred_met_bf['metabolite']]
pred_met_bf['feature'] = [acc_name[m].split(';')[0] for m in pred_met_bf['feature']]


# ---- Metabolites predictions correlations
order = list(pred_met[pred_met['feature'] == 'tf'].sort('cor', ascending=False)['metabolite'])
palette = {'tf': '#34495e', 'kinase': '#3498db'}

sns.set(style='ticks')
fig, gs = plt.figure(figsize=(10, 15)), GridSpec(5, 2, width_ratios=[1, 1], hspace=0.4)

# Metabolites prediction plot
# Barplot
ax = plt.subplot(gs[:, 0])
sns.barplot('cor', 'metabolite', 'feature', pred_met, palette=palette, ci=None, orient='h', lw=0, order=order, ax=ax)
ax.axvline(x=0, ls=':', c='.5')
ax.set_xlabel('Pearson correlation')
ax.set_ylabel('Metabolite')
ax.set_title('Predicted vs Measured')
ax.set_xlim((0, 1.0))
sns.despine(ax=ax)

# Scatter
pos = 1
for m in set(pred_met_bf['metabolite']):
    ax = plt.subplot(gs[pos])

    plot_df = pred_met_bf[pred_met_bf['metabolite'] == m]
    best_k = list(set(plot_df.ix[plot_df['feature_type'] == 'kinase', 'feature']))[0]
    best_t = list(set(plot_df.ix[plot_df['feature_type'] == 'tf', 'feature']))[0]

    sns.regplot('x', 'y', data=plot_df[plot_df['feature_type'] == 'kinase'], scatter_kws={'s': 50, 'alpha': .6}, color=palette['kinase'], ax=ax, label=best_k)
    sns.regplot('x', 'y', data=plot_df[plot_df['feature_type'] == 'tf'], scatter_kws={'s': 50, 'alpha': .6}, color=palette['tf'], ax=ax, label=best_t)
    sns.despine(ax=ax)
    ax.set_xlabel(m)
    ax.set_ylabel('Kinase/TF activity')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    pos += 2

plt.savefig('%s/reports/Figure_3.pdf' % wd, bbox_inches='tight')
plt.close('all')

# Feature network plot
beta_thres = 1.5

network = lm_betas[lm_betas['beta'].abs() > beta_thres]
network = network[[m in m_map for m in network['metabolite']]]
network['metabolite'] = [str(m) for m in network['metabolite']]

network_i = igraph.Graph(directed=False)
network_i.add_vertices(list(set(network['metabolite']).union(network['feature'])))
network_i.add_edges([(m, p) for m, p in network[['metabolite', 'feature']].values])
network_i = network_i.simplify()
print '[INFO] Network: ', network_i.summary()


# Set nodes attributes
node_name = lambda x: acc_name[x].split(';')[0] if x in k_activity_dyn.index or x in tf_activity_dyn.index else m_map[float(x)]
network_i.vs['label'] = [node_name(v) for v in network_i.vs['name']]

node_shape = lambda x: 'square' if (x not in k_activity_dyn.index) and (x not in tf_activity_dyn.index) else 'circle'
network_i.vs['shape'] = [node_shape(v) for v in network_i.vs['name']]

node_colour = lambda x: '#3498db' if x in k_activity_dyn.index else ('#34495e' if x in tf_activity_dyn.index else '#e74c3c')
network_i.vs['color'] = [node_colour(v) for v in network_i.vs['name']]

node_label_color = lambda x: 'white' if (x in k_activity_dyn.index) or (x in tf_activity_dyn.index) else 'black'
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
    target='%s/reports/Figure_4.pdf' % wd
)
print '[INFO] Network exported: ', network_i.summary()


# Supplementary materials figures
plot_df = lm_betas_kinase.loc[:, [m in m_map for m in lm_betas_kinase]]
plot_df.columns = [m_map[m] for m in plot_df]
plot_df.index = [acc_name[i].split(';')[0] for i in plot_df.index]

sns.set(style='white', palette='pastel')
cmap, lw = sns.diverging_palette(220, 10, sep=80, n=9, as_cmap=True), .5
sns.clustermap(plot_df.T, figsize=(15, 20), robust=True, cmap=cmap, linewidth=lw)
plt.savefig('%s/reports/Figure_Supp_4_kinases_betas.pdf' % wd, bbox_inches='tight')
plt.close('all')


plot_df = lm_betas_tf.loc[:, [m in m_map for m in lm_betas_tf]]
plot_df.columns = [m_map[m] for m in plot_df]
plot_df.index = [acc_name[i].split(';')[0] for i in plot_df.index]
plot_df = plot_df[plot_df.std(1) != 0]

sns.set(style='white', palette='pastel')
cmap, lw = sns.diverging_palette(220, 10, sep=80, n=9, as_cmap=True), .5
sns.clustermap(plot_df.T, figsize=(15, 20), robust=True, cmap=cmap, linewidth=lw)
plt.savefig('%s/reports/Figure_Supp_4_transcription_factors_betas.pdf' % wd, bbox_inches='tight')
plt.close('all')
