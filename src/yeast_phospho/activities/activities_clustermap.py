import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas import DataFrame, Series, read_csv
from yeast_phospho.utilities import get_proteins_name


# -- Import IDs maps
acc_name = get_proteins_name()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}


# -- Kinases activities Nitrogen metabolism Kinases clustermap
k_activity_dyn_ng_gsea = read_csv('%s/tables/kinase_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_ng_gsea = k_activity_dyn_ng_gsea[(k_activity_dyn_ng_gsea.count(1) / k_activity_dyn_ng_gsea.shape[1]) > .75].replace(np.NaN, 0.0)
k_activity_dyn_ng_gsea.index = [acc_name[i] for i in k_activity_dyn_ng_gsea.index]
print '[INFO] Nitrogen kinases activities: ', k_activity_dyn_ng_gsea.shape

cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)
sns.set(context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.clustermap(k_activity_dyn_ng_gsea.T.corr(), figsize=(5, 5), linewidth=.5, cmap=cmap, metric='correlation')
plt.title('Nitrogen metabolism\n(pearson)')
plt.savefig('%s/reports/kactivities_clustermap_nitrogen_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'

# -- Kinases activities Salt+Pheromone Kinases clustermap
k_activity_dyn_comb_ng = read_csv('%s/tables/kinase_activity_dynamic_combination_gsea.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_comb_ng = k_activity_dyn_comb_ng[[c for c in k_activity_dyn_comb_ng if not c.startswith('NaCl+alpha_')]]
k_activity_dyn_comb_ng = k_activity_dyn_comb_ng[(k_activity_dyn_comb_ng.count(1) / k_activity_dyn_comb_ng.shape[1]) > .75].replace(np.NaN, 0.0)
k_activity_dyn_comb_ng.index = [acc_name[i] for i in k_activity_dyn_comb_ng.index]
print '[INFO] Salt+pheromone kinases activities: ', k_activity_dyn_comb_ng.shape

cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)
sns.set(context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.clustermap(k_activity_dyn_comb_ng.T.corr(), figsize=(5, 5), linewidth=.5, cmap=cmap, metric='correlation')
plt.title('NaCl/Pheromone\n(pearson)')
plt.savefig('%s/reports/kactivities_clustermap_salt-pheromone_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'


# -- Transcription-factors activities Nitrogen metabolism clustermap
tf_activity_dyn_ng_gsea = read_csv('%s/tables/tf_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
tf_activity_dyn_ng_gsea.index = [acc_name[i] for i in tf_activity_dyn_ng_gsea.index]
print '[INFO] Salt+pheromone transcription-factor activities: ', tf_activity_dyn_ng_gsea.shape

cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)
sns.set(context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.clustermap(tf_activity_dyn_ng_gsea.T.corr(), figsize=(14, 14), linewidth=.5, cmap=cmap, metric='correlation')
plt.title('Nitrogen metabolism\n(pearson)')
plt.savefig('%s/reports/tfactivities_clustermap_nitrogen_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'


tf_activity_dyn_gsea = read_csv('%s/tables/tf_activity_dynamic_gsea.tab' % wd, sep='\t', index_col=0)
tf_activity_dyn_gsea.index = [acc_name[i] for i in tf_activity_dyn_gsea.index]
print '[INFO] Salt+pheromone transcription-factor activities: ', tf_activity_dyn_gsea.shape

cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)
sns.set(context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.clustermap(tf_activity_dyn_gsea.T.corr(), figsize=(14, 14), linewidth=.5, cmap=cmap, metric='correlation')
plt.title('Nitrogen metabolism\n(pearson)')
plt.savefig('%s/reports/tfactivities_clustermap_nitrogen_not_normalised_gsea.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'
