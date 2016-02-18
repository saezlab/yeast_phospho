import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from yeast_phospho import wd
from pandas import read_csv
from yeast_phospho.utilities import get_proteins_name


# -- Import IDs maps
acc_name = get_proteins_name()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}


dyn_xorder = [
    'N_downshift_5min', 'N_downshift_9min', 'N_downshift_15min', 'N_downshift_25min', 'N_downshift_44min', 'N_downshift_79min',
    'N_upshift_5min', 'N_upshift_9min', 'N_upshift_15min', 'N_upshift_25min', 'N_upshift_44min', 'N_upshift_79min',
    'Rapamycin_5min', 'Rapamycin_9min', 'Rapamycin_15min', 'Rapamycin_25min', 'Rapamycin_44min', 'Rapamycin_79min'
]


# -- Import
# Steady-state without growth
k_activity_ng = read_csv('%s/tables/kinase_activity_steady_state_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_ng = k_activity_ng[(k_activity_ng.count(1) / k_activity_ng.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_ng = read_csv('%s/tables/tf_activity_steady_state_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
tf_activity_ng = tf_activity_ng[tf_activity_ng.std(1) > .4]


# Dynamic without growth
k_activity_dyn_ng = read_csv('%s/tables/kinase_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_ng = k_activity_dyn_ng[(k_activity_dyn_ng.count(1) / k_activity_dyn_ng.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_dyn_ng = read_csv('%s/tables/tf_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
tf_activity_dyn_ng = tf_activity_dyn_ng[tf_activity_dyn_ng.std(1) > .4]


# -- Plot
sns.set(style='white')

# Heatmaps
fig, gs = plt.figure(figsize=(15, 25)), GridSpec(2, 2, width_ratios=[2.5, 1], height_ratios=[.25, .5])
cbar_ax = plt.subplot(gs[2])
cbar_ax.set_title('Activity')

cmap, lw = sns.diverging_palette(220, 10, n=9, as_cmap=True), .5

ax00 = plt.subplot(gs[0])
plot_df = k_activity_ng.copy().replace(np.NaN, 0.0)
plot_df.columns = [acc_name[i] for i in plot_df.columns]
plot_df.index = [acc_name[i] for i in plot_df.index]
plot_df.index.name = 'Kinases'
sns.heatmap(plot_df, ax=ax00, robust=True, cbar_ax=cbar_ax, linewidths=lw, cmap=cmap)
ax00.set_title('Steady-state')
plt.setp(ax00.get_xticklabels(), visible=False)
print '[INFO] Clustemap done!'

ax01 = plt.subplot(gs[1])
plot_df = k_activity_dyn_ng[dyn_xorder].replace(np.NaN, 0.0)
plot_df.index = [acc_name[i] for i in plot_df.index]
sns.heatmap(plot_df, ax=ax01, robust=True, cbar=False, linewidths=lw, cmap=cmap)
ax01.set_title('Dynamic')
plt.setp(ax01.get_xticklabels(), visible=False)
print '[INFO] Clustemap done!'

ax10 = plt.subplot(gs[2], sharex=ax00)
plot_df = tf_activity_ng.copy().replace(np.NaN, 0.0)
plot_df.columns = [acc_name[i] for i in plot_df.columns]
plot_df.index = [acc_name[i] for i in plot_df.index]
plot_df.columns.name, plot_df.index.name = 'Perturbations', 'Transcription factors'
sns.heatmap(plot_df, ax=ax10, robust=True, cbar=False, linewidths=lw, cmap=cmap)
print '[INFO] Clustemap done!'

ax11 = plt.subplot(gs[3], sharex=ax01)
plot_df = tf_activity_dyn_ng[dyn_xorder].replace(np.NaN, 0.0)
plot_df.index = [acc_name[i] for i in plot_df.index]
plot_df.columns.name = 'Dynamic conditions'
sns.heatmap(plot_df, ax=ax11, robust=True, cbar=False, linewidths=lw, cmap=cmap)
print '[INFO] Clustemap done!'

# Export figure
fig.tight_layout()
plt.savefig('%s/reports/Figure_1.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'
