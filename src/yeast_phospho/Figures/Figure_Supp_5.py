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
# Steady-state
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)
metabolomics = metabolomics[metabolomics.std(1) > .4]

k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)
k_activity = k_activity[(k_activity.count(1) / k_activity.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0)


# ---- Perform predictions
# Steady-state comparisons
steady_state = [
    (k_activity.copy(), metabolomics.copy(), 'kinase'),
    (tf_activity.copy(), metabolomics.copy(), 'tf'),
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


# Supplementary materials figures
plot_df = lm_betas_kinase.loc[:, [m in m_map for m in lm_betas_kinase]]
plot_df.columns = [m_map[m] for m in plot_df]
plot_df.index = [acc_name[i].split(';')[0] for i in plot_df.index]

sns.set(style='white', palette='pastel')
cmap, lw = sns.diverging_palette(220, 10, n=9, as_cmap=True), .5
sns.clustermap(plot_df.T, figsize=(15, 20), robust=True, cmap=cmap, linewidth=lw)
plt.savefig('%s/reports/Figure_Supp_5_kinases_betas_steadystate.pdf' % wd, bbox_inches='tight')
plt.close('all')


plot_df = lm_betas_tf.loc[:, [m in m_map for m in lm_betas_tf]]
plot_df.columns = [m_map[m] for m in plot_df]
plot_df.index = [acc_name[i].split(';')[0] for i in plot_df.index]
plot_df = plot_df[plot_df.std(1) != 0]

sns.set(style='white', palette='pastel')
cmap, lw = sns.diverging_palette(220, 10, n=9, as_cmap=True), .5
sns.clustermap(plot_df.T, figsize=(15, 20), robust=True, cmap=cmap, linewidth=lw)
plt.savefig('%s/reports/Figure_Supp_5_transcription_factors_betas_steadystate.pdf' % wd, bbox_inches='tight')
plt.close('all')

print '[INFO] Done'
