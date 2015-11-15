import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from yeast_phospho import wd
from pandas import read_csv, DataFrame, melt

# Import id maps
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)['gene'].to_dict()


# ---- Import
# Steady-state
k_activity = read_csv('%s/tables/kinase_activity_steady_state_no_growth.tab' % wd, sep='\t', index_col=0)
tf_activity = read_csv('%s/tables/tf_activity_steady_state_no_growth.tab' % wd, sep='\t', index_col=0).dropna()

strains = list(set(k_activity.columns).intersection(tf_activity.columns))
k_activity, tf_activity = k_activity[strains], tf_activity[strains]

# k_activity = k_activity[(k_activity.abs() > 1).sum(1) > 2]
tf_activity = tf_activity[(tf_activity.abs() > 1).sum(1) > 2]

# Dynamic
dyn_xorder = [
    'N_downshift_5min', 'N_downshift_9min', 'N_downshift_15min', 'N_downshift_25min', 'N_downshift_44min', 'N_downshift_79min',
    'N_upshift_5min', 'N_upshift_9min', 'N_upshift_15min', 'N_upshift_25min', 'N_upshift_44min', 'N_upshift_79min',
    'Rapamycin_5min', 'Rapamycin_9min', 'Rapamycin_15min', 'Rapamycin_25min', 'Rapamycin_44min', 'Rapamycin_79min'
]

k_activity_dyn = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0)
tf_activity_dyn = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0).dropna()

# k_activity_dyn = k_activity_dyn[(k_activity_dyn.abs() > 1).sum(1) > 2]
tf_activity_dyn = tf_activity_dyn[(tf_activity_dyn.abs() > 1).sum(1) > 2]

k_activity_dyn, tf_activity_dyn = k_activity_dyn[dyn_xorder], tf_activity_dyn[dyn_xorder]


# ---- Plot
sns.set(style='white', context='paper')

# Heatmaps
fig, gs = plt.figure(figsize=(15, 25)), GridSpec(2, 5, width_ratios=[2.5, 1, .05, .2, .7], height_ratios=[2.5, 1])
cbar_ax = plt.subplot(gs[2])
cbar_ax.set_title('Activity')

cmap, lw = sns.diverging_palette(220, 10, sep=80, n=9, as_cmap=True), .5

ax00 = plt.subplot(gs[0])
plot_df = k_activity.copy().replace(np.NaN, 0.0)
plot_df.columns = [acc_name[i].split(';')[0] for i in plot_df.columns]
plot_df.index = [acc_name[i].split(';')[0] for i in plot_df.index]
plot_df.index.name = 'Kinases'
sns.heatmap(plot_df, ax=ax00, robust=True, cbar_ax=cbar_ax, linewidths=lw, cmap=cmap)
ax00.set_title('Steady-state')
plt.setp(ax00.get_xticklabels(), visible=False)
print '[INFO] Clustemap done!'

ax01 = plt.subplot(gs[1])
plot_df = k_activity_dyn.copy().replace(np.NaN, 0.0)
plot_df.index = [acc_name[i].split(';')[0] for i in plot_df.index]
sns.heatmap(plot_df, ax=ax01, robust=True, cbar=False, linewidths=lw, cmap=cmap)
ax01.set_title('Dynamic')
plt.setp(ax01.get_xticklabels(), visible=False)
print '[INFO] Clustemap done!'

ax10 = plt.subplot(gs[5], sharex=ax00)
plot_df = tf_activity.copy().replace(np.NaN, 0.0)
plot_df.columns = [acc_name[i].split(';')[0] for i in plot_df.columns]
plot_df.index = [acc_name[i].split(';')[0] for i in plot_df.index]
plot_df.columns.name, plot_df.index.name = 'Perturbations', 'Transcription factors'
sns.heatmap(plot_df, ax=ax10, robust=True, cbar=False, linewidths=lw, cmap=cmap)
print '[INFO] Clustemap done!'

ax11 = plt.subplot(gs[6], sharex=ax01)
plot_df = tf_activity_dyn.copy().replace(np.NaN, 0.0)
plot_df.index = [acc_name[i].split(';')[0] for i in plot_df.index]
plot_df.columns.name = 'Dynamic conditions'
sns.heatmap(plot_df, ax=ax11, robust=True, cbar=False, linewidths=lw, cmap=cmap)
print '[INFO] Clustemap done!'

# Boxplots
ax02 = plt.subplot(gs[:, 4])

gene_annotation = read_csv('%s/files/gene_association.txt' % wd, sep='\t', header=None).dropna(how='all', axis=1)
gene_annotation['gene'] = [i.split('|')[0] if str(i) != 'nan' else '' for i in gene_annotation[10]]
gene_annotation = gene_annotation.groupby('gene').first()

kinases_type = DataFrame([(i, gene_annotation.ix[i, 9]) for i in k_activity.index], columns=['name', 'info'])
kinases_type['type'] = ['Kinase' if 'kinase' in i.lower() else 'Phosphatase' if 'phosphatase' in i.lower() else 'ND' for i in kinases_type['info']]
kinases_type = kinases_type.set_index('name')

plot_df = k_activity.copy()
plot_df.columns.name = 'strain'
plot_df['kinase'] = plot_df.index
plot_df = melt(plot_df, id_vars='kinase', value_name='activity').dropna()
plot_df['type'] = [kinases_type.ix[i, 'type'] for i in plot_df['kinase']]
plot_df['diagonal'] = ['KO' if k == s else 'WT' for k, s in plot_df[['kinase', 'strain']].values]

# sns.boxplot(data=plot_df, x='type', y='activity', hue='diagonal', palette={'KO': '#e74c3c', 'WT': '#95a5a6'}, hue_order=['KO', 'WT'], order=['Kinase', 'Phosphatase'], sym='', ax=ax02)
# sns.stripplot(data=plot_df, x='type', y='activity', hue='diagonal', palette={'KO': '#e74c3c', 'WT': '#95a5a6'}, hue_order=['KO', 'WT'], order=['Kinase', 'Phosphatase'], jitter=True, size=7, ax=ax02)
# ax02.axhline(y=0, ls=':', c='.5')
# ax02.legend().remove()
# ax02.set_ylabel('Activity')
# ax02.set_xlabel('')
# ax02.set_title('Knockouts\nactivity change')
# sns.despine(trim=True, ax=ax02)

# Export figure
fig.tight_layout()
plt.savefig('%s/reports/Figure_1.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'
