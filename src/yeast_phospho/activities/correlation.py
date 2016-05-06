import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas import DataFrame, Series, read_csv, concat


# -- GSEA activities
# Steady-state
k_activity_gsea = read_csv('%s/tables/kinase_activity_steady_state_gsea.tab' % wd, sep='\t', index_col=0)
tf_activity_gsea = read_csv('%s/tables/tf_activity_steady_state_gsea.tab' % wd, sep='\t', index_col=0)

# Dynamic
k_activity_dyn_gsea = read_csv('%s/tables/kinase_activity_dynamic_gsea.tab' % wd, sep='\t', index_col=0)
tf_activity_dyn_gsea = read_csv('%s/tables/tf_activity_dynamic_gsea.tab' % wd, sep='\t', index_col=0)

# Dynamic combination
k_activity_dyn_comb_ng_gsea = read_csv('%s/tables/kinase_activity_dynamic_combination_gsea.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_comb_ng_gsea = k_activity_dyn_comb_ng_gsea[[c for c in k_activity_dyn_comb_ng_gsea if not c.startswith('NaCl+alpha_')]]


# -- LM activities
# Steady-state
k_activity_lm = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)
tf_activity_lm = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0)

# Dynamic
k_activity_dyn_lm = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0)
tf_activity_dyn_lm = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0)

# Dynamic combination
k_activity_dyn_comb_ng_lm = read_csv('%s/tables/kinase_activity_dynamic_combination.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_comb_ng_lm = k_activity_dyn_comb_ng_lm[[c for c in k_activity_dyn_comb_ng_lm if not c.startswith('NaCl+alpha_')]]


# -- Clustermaps
palette = {'Rapamycin': '#D25A2B', 'N_upshift': '#5EACEC', 'N_downshift': '#4783C7', 'NaCl': '#CC2229', 'Pheromone': '#6FB353', 'Genetic perturbations': '#f6eb7f'}
order = ['Genetic perturbations', 'N_downshift', 'N_upshift', 'Rapamycin', 'NaCl', 'Pheromone']

plot_df = DataFrame(concat([df1.corrwith(df2) for df1, df2 in [(k_activity_gsea, k_activity_lm), (k_activity_dyn_gsea, k_activity_dyn_lm), (k_activity_dyn_comb_ng_gsea, k_activity_dyn_comb_ng_lm)]])).reset_index()
plot_df.columns = ['condition', 'pearson']
plot_df['Experiment'] = ['Genetic perturbations' if len(c.split('_')[:-1]) == 0 else '_'.join(c.split('_')[:-1]).replace('alpha', 'Pheromone') for c in plot_df['condition']]
plot_df = plot_df.sort('pearson', ascending=False)

sns.set(style='ticks', context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
g = sns.factorplot(
    y='condition', x='pearson', hue='Experiment', data=plot_df, kind='point', lw=0, palette=palette, legend_out=True, aspect=.2, size=12., ci=None, split=False, hue_order=order
)
g.set(xlim=(0, 1))
g.despine(trim=True)
g.set_axis_labels('Activities correlation\n(pearson)', 'Conditions')
g.fig.subplots_adjust(wspace=.2, hspace=.2)
plt.savefig('%s/reports/k_activities_correlation.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'


# -- Scatter
best_cor, worst_cor = 'YGL021W', 'N_upshift_25min'
color = '#808080'

# Best
x = k_activity_gsea[best_cor].dropna()
y = k_activity_lm.ix[x.index, best_cor]

sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3}, font_scale=1.5)
g = sns.jointplot(
    x, y, kind='reg', color=color, joint_kws={'scatter_kws': {'s': 40, 'edgecolor': 'w', 'linewidth': .5}},
    marginal_kws={'hist': False, 'rug': True}, space=0,
)
plt.axhline(0, ls='-', lw=0.3, c=color, alpha=.5)
plt.axvline(0, ls='-', lw=0.3, c=color, alpha=.5)
g.plot_marginals(sns.kdeplot, shade=True, color=color)
g.set_axis_labels('Kinase activity (GSEA)', 'Kinase activity (Ridge)')
plt.savefig('%s/reports/k_activities_correlation_best.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Corr plotted!'

# Worst
x = k_activity_dyn_gsea[worst_cor].dropna()
y = k_activity_dyn_lm.ix[x.index, worst_cor]

sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3}, font_scale=1.5)
g = sns.jointplot(
    x, y, kind='reg', color=color, joint_kws={'scatter_kws': {'s': 40, 'edgecolor': 'w', 'linewidth': .5}},
    marginal_kws={'hist': False, 'rug': True}, space=0,
)
plt.axhline(0, ls='-', lw=0.3, c=color, alpha=.5)
plt.axvline(0, ls='-', lw=0.3, c=color, alpha=.5)
g.plot_marginals(sns.kdeplot, shade=True, color=color)
g.set_axis_labels('Kinase activity (GSEA)', 'Kinase activity (Ridge)')
plt.savefig('%s/reports/k_activities_correlation_worst.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Corr plotted!'
