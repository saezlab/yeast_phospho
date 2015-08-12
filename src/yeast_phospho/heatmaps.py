import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas import read_csv

# Import id maps
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)['gene'].to_dict()


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
metabolomics_dyn = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0).ix[m_signif].dropna()
k_activity_dyn = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0).ix[k_signif].dropna(how='all').replace(np.NaN, 0.0)
tf_activity_dyn = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0).dropna()


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


# ---- Kinase activities heatmaps
plot_df = k_activity.copy().replace(np.NaN, 0.0)
plot_df.columns = [acc_name[i].split(';')[0] for i in plot_df.columns]
plot_df.index = [acc_name[i].split(';')[0] for i in plot_df.index]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'kinases'
sns.clustermap(plot_df, figsize=(10, 10), linewidths=0)
plt.savefig(wd + 'reports/heatmap_kinome_activity_steady_state.pdf', bbox_inches='tight')
plt.close('all')
print '[INFO] Clustemap done!'


plot_df = k_activity_dyn.copy().replace(np.NaN, 0.0)
plot_df.index = [acc_name[i].split(';')[0] for i in plot_df.index]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'kinases'
sns.clustermap(plot_df, figsize=(8, 10), linewidths=0)
plt.savefig(wd + 'reports/heatmap_kinome_activity_dynamic.pdf', bbox_inches='tight')
plt.close('all')
print '[INFO] Clustemap done!'


# ---- Transcription factors heatmaps
plot_df = tf_activity.copy().replace(np.NaN, 0.0)
plot_df = plot_df[plot_df.std(1) > .2]
plot_df.columns = [acc_name[i].split(';')[0] for i in plot_df.columns]
plot_df.index = [acc_name[i].split(';')[0] for i in plot_df.index]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'transcription factors'
sns.clustermap(plot_df, figsize=(10, 15), linewidths=0)
plt.savefig(wd + 'reports/heatmap_transcriptome_activity_steady_state.pdf', bbox_inches='tight')
plt.close('all')
print '[INFO] Clustemap done!'


plot_df = tf_activity_dyn.copy().replace(np.NaN, 0.0)
plot_df = plot_df[plot_df.std(1) > .2]
plot_df.index = [acc_name[i].split(';')[0] for i in plot_df.index]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'transcription factors'
g = sns.clustermap(plot_df, figsize=(8, 10), linewidths=0)
plt.savefig(wd + 'reports/heatmap_transcriptome_activity_dynamic.pdf', bbox_inches='tight')
plt.close('all')
print '[INFO] Clustemap done!'
