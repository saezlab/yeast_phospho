import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas import read_csv

# Import id maps
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)

# ---- Metabolomics heatmaps
metabolomics_growth = read_csv('%s/tables/metabolomics_steady_state_growth_rate.tab' % wd, sep='\t', index_col=0)
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)

metabolites, strains = list(metabolomics.index), list(metabolomics.columns)
metabolomics, metabolomics_growth = metabolomics.ix[metabolites, strains], metabolomics_growth.ix[metabolites, strains]

plot_df = metabolomics.copy()
plot_df.columns = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.columns]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'metabolites'
sns.clustermap(plot_df, figsize=(25, 20), yticklabels=False, linewidths=0)
plt.savefig(wd + 'reports/heatmap_metabolomics_steady_state.png', bbox_inches='tight')
plt.close('all')

plot_df = metabolomics_growth.copy()
plot_df.columns = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.columns]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'metabolites'
sns.clustermap(plot_df, figsize=(25, 20), yticklabels=False, linewidths=0)
plt.savefig(wd + 'reports/heatmap_metabolomics_steady_state_with_growth.png', bbox_inches='tight')
plt.close('all')
print '[INFO] Clustemap done!'


# ---- Kinase activities heatmaps
k_activity_growth = read_csv('%s/tables/kinase_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0)
k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)

plot_df = k_activity.copy().replace(np.NaN, 0.0)
plot_df.columns = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.columns]
plot_df.index = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.index]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'kinases'
sns.clustermap(plot_df, figsize=(25, 25), linewidths=0)
plt.savefig(wd + 'reports/heatmap_kinome_activity_steady_state.png', bbox_inches='tight')
plt.close('all')

plot_df = k_activity_growth.copy().replace(np.NaN, 0.0)
plot_df.columns = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.columns]
plot_df.index = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.index]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'kinases'
sns.clustermap(plot_df, figsize=(25, 25), linewidths=0)
plt.savefig(wd + 'reports/heatmap_kinome_activity_steady_state_growth.png', bbox_inches='tight')
plt.close('all')
print '[INFO] Clustemap done!'

axis_order = list(set(k_activity.index).intersection(k_activity.columns))

plot_df = k_activity.copy().replace(np.NaN, 0.0).loc[axis_order, axis_order]
plot_df.columns = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.columns]
plot_df.index = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.index]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'kinases'
sns.clustermap(plot_df, figsize=(25, 25), linewidths=0, row_cluster=False, col_cluster=False)
plt.savefig(wd + 'reports/heatmap_kinome_activity_steady_state_diagonal.png', bbox_inches='tight')
plt.close('all')

plot_df = k_activity_growth.copy().replace(np.NaN, 0.0).loc[axis_order, axis_order]
plot_df.columns = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.columns]
plot_df.index = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.index]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'kinases'
sns.clustermap(plot_df, figsize=(25, 25), linewidths=0, row_cluster=False, col_cluster=False)
plt.savefig(wd + 'reports/heatmap_kinome_activity_steady_state_diagonal_growth.png', bbox_inches='tight')
plt.close('all')
print '[INFO] Clustemap diagonal done!'


# ---- Transcription factors heatmaps
tf_activity_growth = read_csv('%s/tables/tf_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0)
tf_activity = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0)

plot_df = tf_activity.copy().replace(np.NaN, 0.0)
plot_df.columns = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.columns]
plot_df.index = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.index]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'transcription factors'
sns.clustermap(plot_df, figsize=(25, 25), linewidths=0)
plt.savefig(wd + 'reports/heatmap_transcriptome_activity_steady_state.png', bbox_inches='tight')
plt.close('all')

plot_df = tf_activity_growth.copy().replace(np.NaN, 0.0)
plot_df.columns = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.columns]
plot_df.index = [acc_name.loc[i, 'gene'].split(';')[0] for i in plot_df.index]
plot_df.columns.name, plot_df.index.name = 'perturbations', 'transcription factors'
sns.clustermap(plot_df, figsize=(25, 25), linewidths=0)
plt.savefig(wd + 'reports/heatmap_transcriptome_activity_steady_state_growth.png', bbox_inches='tight')
plt.close('all')
print '[INFO] Clustemap done!'
