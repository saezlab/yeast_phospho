import igraph
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from matplotlib.gridspec import GridSpec
from pandas import DataFrame, read_csv, melt, concat
from yeast_phospho.utilities import get_proteins_name, get_metabolites_name


# -- Import IDs maps
acc_name = get_proteins_name()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}

met_name = get_metabolites_name()
met_name = {k: met_name[k] for k in met_name if len(met_name[k].split('; ')) == 1}


# -- Import
# Dynamic data-sets
metabolomics = read_csv('%s/tables/metabolomics_dynamic_no_growth.tab' % wd, sep='\t', index_col=0).dropna()
metabolomics.index = ['%.4f' % i for i in metabolomics.index]
metabolomics = metabolomics[metabolomics.std(1) > .4]

k_activity = read_csv('%s/tables/kinase_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity = k_activity[(k_activity.count(1) / k_activity.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity = read_csv('%s/tables/tf_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0).replace(np.NaN, 0.0)
tf_activity = tf_activity[tf_activity.std(1) > .4]


# Linear regression results
with open('%s/tables/linear_regressions.pickle' % wd, 'rb') as handle:
    lm_res = pickle.load(handle)


lm_betas_kinases = DataFrame([i[1][3] for i in lm_res if i[1][0] == 'Kinases' and i[1][1] == 'Dynamic' and i[1][2] == 'without'][0])
lm_betas_tfs = DataFrame([i[1][3] for i in lm_res if i[1][0] == 'TFs' and i[1][1] == 'Dynamic' and i[1][2] == 'without'][0])

lm_cor = [(ft, dt, f, mt, ct, c) for c in lm_res for ft, dt, f, mt, ct, c in c[0]]
lm_cor = DataFrame(lm_cor, columns=['feature', 'dataset', 'variable', 'growth', 'corr_type', 'cor'])
print '[INFO] Data-sets + Linear regression results imported'


# -- Plot
palette = {'TFs': '#34495e', 'Kinases': '#3498db'}

plot_df = lm_cor[(lm_cor['growth'] == 'without') & (lm_cor['dataset'] == 'Dynamic')]
plot_df = plot_df[[i in met_name for i in plot_df['variable']]]
plot_df['metabolite'] = [met_name[i] for i in plot_df['variable']]

order = list(plot_df[plot_df['feature'] == 'TFs'].sort('cor', ascending=False)['metabolite'])


# Barplot
sns.set(style='ticks')
fig, gs = plt.figure(figsize=(10, 15)), GridSpec(4, 2, width_ratios=[1, 1], hspace=0.45, wspace=0.3)

ax = plt.subplot(gs[:, 0])
sns.barplot('cor', 'metabolite', 'feature', plot_df, palette=palette, ci=None, orient='h', lw=0, order=order, ax=ax)
ax.axvline(x=0, ls=':', c='.5')
ax.set_xlabel('Pearson correlation')
ax.set_ylabel('Metabolite')
ax.set_title('Predicted vs Measured')
ax.set_xlim((0, 1.0))
sns.despine(ax=ax)


# Scatter
pos = 1
for m in ['91.0400', '171.0100', '115.0000', '104.0400']:
    ax = plt.subplot(gs[pos])

    best_tf = lm_betas_tfs[m].abs().argmax()
    best_kinase = lm_betas_kinases[m].abs().argmax()

    sns.regplot(metabolomics.ix[m], k_activity.ix[best_kinase], scatter_kws={'s': 50, 'alpha': .6}, color=palette['Kinases'], label=acc_name[best_kinase], ax=ax)
    sns.regplot(metabolomics.ix[m], tf_activity.ix[best_tf], scatter_kws={'s': 50, 'alpha': .6}, color=palette['TFs'], label=acc_name[best_tf], ax=ax)
    sns.despine(ax=ax)
    ax.set_title('%s' % met_name[m])
    ax.set_xlabel('Metabolite (log FC)')
    ax.set_ylabel('Kinase/TF activity')
    ax.axhline(0, ls='--', c='.5', lw=.3)
    ax.axvline(0, ls='--', c='.5', lw=.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    pos += 2

plt.savefig('%s/reports/Figure_3.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Figure 3 exported'


# -- Figure 4
def flatten_betas(df, ftype):
    lm_betas = df.copy()
    lm_betas['feature'] = lm_betas.index
    lm_betas = melt(lm_betas, id_vars='feature')
    lm_betas['type'] = ftype
    return lm_betas

network = concat([flatten_betas(df, ftype) for df, ftype in [(lm_betas_tfs, 'TFs'), (lm_betas_kinases, 'Kinases')]])
network = network[[m in met_name for m in network['variable']]]
network = network[network['value'].abs() > .4]


network_i = igraph.Graph(directed=False)
network_i.add_vertices(list(set(network['variable']).union(network['feature'])))
network_i.add_edges([(m, p) for m, p in network[['variable', 'feature']].values])
network_i.es['beta'] = [v for v in network['value']]
print '[INFO] Network: ', network_i.summary()


# Set nodes attributes
node_name = lambda x: acc_name[x].split(';')[0] if x in k_activity.index or x in tf_activity.index else met_name[x]
network_i.vs['label'] = [node_name(v) for v in network_i.vs['name']]

node_shape = lambda x: 'square' if (x not in k_activity.index) and (x not in tf_activity.index) else 'circle'
network_i.vs['shape'] = [node_shape(v) for v in network_i.vs['name']]

node_colour = lambda x: '#3498db' if x in k_activity.index else ('#34495e' if x in tf_activity.index else '#e74c3c')
network_i.vs['color'] = [node_colour(v) for v in network_i.vs['name']]

node_label_color = lambda x: 'white' if (x in k_activity.index) or (x in tf_activity.index) else 'black'
network_i.vs['label_color'] = [node_label_color(v) for v in network_i.vs['name']]

# Set edges attributes
network_i.es['color'] = ['#e74c3c' if e['beta'] < 0 else '#2ecc71' for e in network_i.es]

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


# -- Betas heatmap
cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)

plot_df = lm_betas_kinases.loc[:, [m in met_name for m in lm_betas_kinases]]
plot_df.columns = [met_name[m] for m in plot_df]
plot_df.index = [acc_name[i].split(';')[0] for i in plot_df.index]

sns.set(style='white', palette='pastel')
sns.clustermap(plot_df.T, figsize=(15, 20), cmap=cmap, linewidth=.5)
plt.savefig('%s/reports/Figure_Supp_4_kinases_dynamic_betas.pdf' % wd, bbox_inches='tight')
plt.close('all')


plot_df = lm_betas_tfs.loc[:, [m in met_name for m in lm_betas_tfs]]
plot_df.columns = [met_name[m] for m in plot_df]
plot_df.index = [acc_name[i].split(';')[0] for i in plot_df.index]
plot_df = plot_df[plot_df.std(1) != 0]

sns.set(style='white', palette='pastel')
sns.clustermap(plot_df.T, figsize=(15, 20), cmap=cmap, linewidth=.5)
plt.savefig('%s/reports/Figure_Supp_4_transcription_factors_dynamic_betas.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Betas heatmaps exported'
