import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas import DataFrame, read_csv
from pymist.enrichment.gsea import gsea
from yeast_phospho.utilities import get_kinases_targets
from yeast_phospho.utilities import get_proteins_name


# -- Import IDs maps
acc_name = get_proteins_name()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}


# -- Dynamic nitrogen TFs
tf_activity_dyn = read_csv('%s/tables/tf_activity_dynamic_gsea.tab' % wd, sep='\t', index_col=0)
tf_activity_dyn['Rapamycin_0min'] = 0
tf_activity_dyn['N_upshift_0min'] = 0
tf_activity_dyn['N_downshift_0min'] = 0

tf_activity_dyn = tf_activity_dyn.unstack().reset_index()
tf_activity_dyn.columns = ['condition', 'tf', 'activity']
tf_activity_dyn['time'] = [int(i.split('_')[-1:][0].replace('min', '')) for i in tf_activity_dyn['condition']]
tf_activity_dyn['stimulation'] = [i.split('_')[0] if i.startswith('Rapamycin') else '_'.join(i.split('_')[:2]) for i in tf_activity_dyn['condition']]
tf_activity_dyn['unit'] = 0
tf_activity_dyn['tf_name'] = [acc_name[k] for k in tf_activity_dyn['tf']]

# Plot
palette = {'Rapamycin': '#D25A2B', 'N_upshift': '#5EACEC', 'N_downshift': '#4783C7'}

features = ['YKL062W', 'YMR037C', 'YER040W', 'YBR083W']

plot_df = tf_activity_dyn[[k in features for k in tf_activity_dyn['tf']]].dropna()

sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3}, font_scale=.75)
g = sns.FacetGrid(plot_df, col='tf_name', sharex=False, sharey=False, size=1.5, aspect=1, legend_out=True, col_order=[acc_name[p] for p in features], ylim=[-3.2, 3.2])
g.map_dataframe(sns.tsplot, time='time', unit='unit', condition='stimulation', value='activity', color=palette, marker='o', lw=.3)
g.map(plt.axhline, y=0, ls='-', lw=0.3, c='black', alpha=.5)
g.set_titles(col_template='{col_name}')
g.set_axis_labels('Time (minutes)', 'Activity')
g.add_legend()
plt.savefig('%s/reports/tf_activities_dynamic_nitrogen_tsplot.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Corr plotted!'


# -- Dynamic nitrogen
# Import
k_activity_dyn = read_csv('%s/tables/kinase_activity_dynamic_gsea.tab' % wd, sep='\t', index_col=0)
k_activity_dyn['Rapamycin_0min'] = 0
k_activity_dyn['N_upshift_0min'] = 0
k_activity_dyn['N_downshift_0min'] = 0

k_activity_dyn = k_activity_dyn.unstack().reset_index()
k_activity_dyn.columns = ['condition', 'kinase', 'activity']
k_activity_dyn['time'] = [int(i.split('_')[-1:][0].replace('min', '')) for i in k_activity_dyn['condition']]
k_activity_dyn['stimulation'] = [i.split('_')[0] if i.startswith('Rapamycin') else '_'.join(i.split('_')[:2]) for i in k_activity_dyn['condition']]
k_activity_dyn['unit'] = 0
k_activity_dyn['kinase_name'] = [acc_name[k] for k in k_activity_dyn['kinase']]

# Plot
palette = {'Rapamycin': '#D25A2B', 'N_upshift': '#5EACEC', 'N_downshift': '#4783C7'}

features = ['YNL183C', 'YFL033C', 'YJL141C']
phosphatases = []

plot_df = k_activity_dyn[[k in features for k in k_activity_dyn['kinase']]].dropna()
plot_df['type'] = ['Phosphatase' if k in phosphatases else 'Kinase' for k in plot_df['kinase']]
plot_df = plot_df.drop([i for k in features for c in palette if len(plot_df[(plot_df['kinase'] == k) & (plot_df['stimulation'] == c)]) <= 1 for i in plot_df[(plot_df['kinase'] == k) & (plot_df['stimulation'] == c)].index])


sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3}, font_scale=.75)
g = sns.FacetGrid(plot_df, col='kinase_name', sharex=False, sharey=False, size=1.5, aspect=1, legend_out=True, col_order=[acc_name[p] for p in features])
g.map_dataframe(sns.tsplot, time='time', unit='unit', condition='stimulation', value='activity', color=palette, marker='o', lw=.3)
g.map(plt.axhline, y=0, ls='-', lw=0.3, c='black', alpha=.5)
g.set_titles(col_template='{col_name}')
g.set_axis_labels('Time (minutes)', 'Activity')
g.add_legend()
plt.savefig('%s/reports/k_activities_dynamic_nitrogen_tsplot.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Corr plotted!'


# -- Dynamic combination
# Import
k_activity_comb_dyn = read_csv('%s/tables/kinase_activity_dynamic_combination_gsea.tab' % wd, sep='\t', index_col=0)
k_activity_comb_dyn = k_activity_comb_dyn[[c for c in k_activity_comb_dyn if not c.startswith('NaCl+alpha')]]
k_activity_comb_dyn['alpha_0'] = 0
k_activity_comb_dyn['NaCl_0'] = 0

k_activity_comb_dyn = k_activity_comb_dyn.unstack().reset_index()
k_activity_comb_dyn.columns = ['condition', 'kinase', 'activity']
k_activity_comb_dyn['time'] = [int(i.split('_')[1]) / 60 for i in k_activity_comb_dyn['condition']]
k_activity_comb_dyn['stimulation'] = ['NaCl' if i.split('_')[0] == 'NaCl' else 'Pheromone' for i in k_activity_comb_dyn['condition']]
k_activity_comb_dyn['unit'] = 0
k_activity_comb_dyn['kinase_name'] = [acc_name[k] for k in k_activity_comb_dyn['kinase']]


# Plot
features = ['YDL159W', 'YLR113W', 'YJL128C']
phosphatases = ['YDL006W', 'YER089C', 'YBL056W']

plot_df = k_activity_comb_dyn[[k in features for k in k_activity_comb_dyn['kinase']]].dropna()
plot_df['type'] = ['Phosphatase' if k in phosphatases else 'Kinase' for k in plot_df['kinase']]

palette = {'NaCl': '#CC2229', 'Pheromone': '#6FB353'}

sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3}, font_scale=.75)
g = sns.FacetGrid(plot_df, col='kinase_name', sharex=False, sharey=False, size=1.5, aspect=1, legend_out=True, col_order=[acc_name[p] for p in features])
g.map_dataframe(sns.tsplot, time='time', unit='unit', condition='stimulation', value='activity', color=palette, marker='o', lw=.3)
g.map(plt.axhline, y=0, ls='-', lw=0.3, c='black', alpha=.5)
g.set_titles(col_template='{col_name}')
g.set_axis_labels('Time (minutes)', 'Activity')
g.add_legend()
plt.savefig('%s/reports/k_activities_dynamic_combination_tsplot.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Corr plotted!'


