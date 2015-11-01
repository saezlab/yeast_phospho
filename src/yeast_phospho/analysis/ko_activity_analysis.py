import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas import DataFrame, melt, read_csv

# Import data-set
k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)

# Import protein annotation
gene_annotation = read_csv('%s/files/gene_association.txt' % wd, sep='\t', header=None).dropna(how='all', axis=1)
gene_annotation['gene'] = [i.split('|')[0] if str(i) != 'nan' else '' for i in gene_annotation[10]]
gene_annotation = gene_annotation.groupby('gene').first()

kinases_type = DataFrame([(i, gene_annotation.ix[i, 9]) for i in k_activity.index], columns=['name', 'info'])
kinases_type['type'] = ['Kinase' if 'kinase' in i.lower() else 'Phosphatase' if 'phosphatase' in i.lower() else 'ND' for i in kinases_type['info']]
kinases_type = kinases_type.set_index('name')

# Plot
plot_df = k_activity.copy()
plot_df.columns.name = 'strain'
plot_df['kinase'] = plot_df.index
plot_df = melt(plot_df, id_vars='kinase', value_name='activity').dropna()
plot_df['type'] = [kinases_type.ix[i, 'type'] for i in plot_df['kinase']]
plot_df['diagonal'] = ['KO' if k == s else 'WT' for k, s in plot_df[['kinase', 'strain']].values]

xorder, hue_order, palette = ['Kinase', 'Phosphatase'], ['KO', 'WT'], {'KO': '#e74c3c', 'WT': '#95a5a6'}

plt.figure(figsize=(1, 3))

sns.set(style='ticks')
sns.boxplot(data=plot_df, x='type', y='activity', hue='diagonal', palette=palette, hue_order=hue_order, order=xorder, sym='')
sns.stripplot(data=plot_df, x='type', y='activity', hue='diagonal', palette=palette, hue_order=hue_order, order=xorder, jitter=True, size=7)
plt.axhline(y=0, ls=':', c='.5')
plt.legend().remove()
plt.ylabel('Activity')
plt.xlabel('')
plt.title('Knockouts\nactivity change')
sns.despine(trim=True)

plt.savefig('%s/reports/kinase_activities_steady_state_boxplot.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Kinases activities ploted'
