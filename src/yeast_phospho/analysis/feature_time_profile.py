import re
import pickle
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as it
from yeast_phospho import wd
from scipy.stats.stats import spearmanr
from pandas.stats.misc import zscore
from statsmodels.api import add_constant
from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import ElasticNetCV, ElasticNet, RidgeCV
from sklearn.metrics.regression import r2_score
from scipy.stats.distributions import hypergeom
from sklearn.cross_validation import ShuffleSplit, LeaveOneOut
from pandas import DataFrame, Series, read_csv, concat, pivot_table
from yeast_phospho.utilities import get_metabolites_name, get_proteins_name


# -- General vars
label_order = ['N_downshift', 'N_upshift', 'Rapamycin', 'NaCl', 'Pheromone']
palette = {'Rapamycin': '#D25A2B', 'N_upshift': '#5EACEC', 'N_downshift': '#4783C7', 'NaCl': '#CC2229', 'Pheromone': '#6FB353'}


# -- Import IDs maps
acc_name = get_proteins_name()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}

met_name = get_metabolites_name()
met_name = {'%.4f' % float(k): met_name[k] for k in met_name if len(met_name[k].split('; ')) == 1}


# -- Import data-sets
# Nitrogen metabolism Metabolomics
metabolomics_dyn_ng = read_csv('%s/tables/metabolomics_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
metabolomics_dyn_ng.index = ['%.4f' % i for i in metabolomics_dyn_ng.index]
metabolomics_dyn_ng = metabolomics_dyn_ng[[i in met_name for i in metabolomics_dyn_ng.index]]
print '[INFO] Nitrogen metabolomics: ', metabolomics_dyn_ng.shape

# Nitrogen metabolism Kinases activities
k_activity_dyn_ng_gsea = read_csv('%s/tables/kinase_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_ng_gsea = k_activity_dyn_ng_gsea[(k_activity_dyn_ng_gsea.count(1) / k_activity_dyn_ng_gsea.shape[1]) > .75].replace(np.NaN, 0.0)
print '[INFO] Nitrogen kinases activities: ', k_activity_dyn_ng_gsea.shape


# Salt+Pheromone Kinases activities
k_activity_dyn_comb_ng = read_csv('%s/tables/kinase_activity_dynamic_combination_gsea.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_comb_ng = k_activity_dyn_comb_ng[[c for c in k_activity_dyn_comb_ng if not c.startswith('NaCl+alpha_')]]
k_activity_dyn_comb_ng = k_activity_dyn_comb_ng[(k_activity_dyn_comb_ng.count(1) / k_activity_dyn_comb_ng.shape[1]) > .75].replace(np.NaN, 0.0)
print '[INFO] Salt+pheromone kinases activities: ', k_activity_dyn_comb_ng.shape

# Salt+Pheromone Metabolomics
metabolomics_dyn_comb = read_csv('%s/tables/metabolomics_dynamic_combination.csv' % wd, index_col=0)[k_activity_dyn_comb_ng.columns]
metabolomics_dyn_comb.index = ['%.4f' % round(i, 2) for i in metabolomics_dyn_comb.index]
metabolomics_dyn_comb = metabolomics_dyn_comb[[i in met_name for i in metabolomics_dyn_comb.index]]
print '[INFO] Salt+pheromone metabolomics: ', metabolomics_dyn_comb.shape

# Replace alpha with pheromone
k_activity_dyn_comb_ng.columns = [c.replace('alpha', 'Pheromone') for c in k_activity_dyn_comb_ng.columns]
metabolomics_dyn_comb.columns = [c.replace('alpha', 'Pheromone') for c in metabolomics_dyn_comb.columns]

k_activity_dyn_comb_ng.columns = ['%s_%.0fmin' % (c.split('_')[0], float(c.split('_')[1]) / 60) for c in k_activity_dyn_comb_ng]
metabolomics_dyn_comb.columns = ['%s_%.0fmin' % (c.split('_')[0], float(c.split('_')[1]) / 60) for c in metabolomics_dyn_comb]


# -- Overlap
ions = list(set(metabolomics_dyn_ng.index).intersection(metabolomics_dyn_comb.index))
kinases = list(set(k_activity_dyn_ng_gsea.index).intersection(k_activity_dyn_comb_ng.index))
conditions = ['Pheromone', 'NaCl', 'N_downshift', 'N_upshift', 'Rapamycin']

ys = concat([metabolomics_dyn_ng.ix[ions], metabolomics_dyn_comb.ix[ions]], axis=1)
xs = concat([k_activity_dyn_ng_gsea.ix[kinases], k_activity_dyn_comb_ng.ix[kinases]], axis=1)


# -- Plot
palette = {'Rapamycin': '#D25A2B', 'N_upshift': '#5EACEC', 'N_downshift': '#4783C7', 'NaCl': '#CC2229', 'Pheromone': '#6FB353'}

# - Plot: kinases activities
xs['Rapamycin_0min'] = 0
xs['N_upshift_0min'] = 0
xs['N_downshift_0min'] = 0
xs['Pheromone_0min'] = 0
xs['NaCl_0min'] = 0

k_activity_dyn = xs.unstack().reset_index()
k_activity_dyn.columns = ['condition', 'kinase', 'activity']
k_activity_dyn['time'] = [int(i.split('_')[-1:][0].replace('min', '')) for i in k_activity_dyn['condition']]
k_activity_dyn['stimulation'] = [i.split('_')[0] if i.split('_')[0] in ['Pheromone', 'NaCl', 'Rapamycin'] else '_'.join(i.split('_')[:2]) for i in k_activity_dyn['condition']]
k_activity_dyn['unit'] = 0
k_activity_dyn['kinase_name'] = [acc_name[k] for k in k_activity_dyn['kinase']]

features = ['YFL033C', 'YJL141C', 'YJL164C', 'YJR066W']
phosphatases = []

plot_df = k_activity_dyn[[k in features for k in k_activity_dyn['kinase']]].dropna()
plot_df['type'] = ['Phosphatase' if k in phosphatases else 'Kinase' for k in plot_df['kinase']]
plot_df = plot_df.drop([i for k in features for c in palette if len(plot_df[(plot_df['kinase'] == k) & (plot_df['stimulation'] == c)]) <= 1 for i in plot_df[(plot_df['kinase'] == k) & (plot_df['stimulation'] == c)].index])

sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3}, font_scale=.75)
g = sns.FacetGrid(plot_df, col='kinase_name', sharex=False, sharey=False, size=1.5, aspect=1, legend_out=True, col_order=[acc_name[p] for p in features])
g.map_dataframe(sns.tsplot, time='time', unit='unit', condition='stimulation', value='activity', color=palette, marker='o', lw=.8)
g.map(plt.axhline, y=0, ls='-', lw=0.3, c='black', alpha=.5)
g.set_titles(col_template='{col_name}')
g.set_axis_labels('Time (minutes)', 'Activity')
g.add_legend()
plt.savefig('%s/reports/tsplot_k_activities.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Corr plotted!'

# -- Plot: metabolite changes
ys['Rapamycin_0min'] = 0
ys['N_upshift_0min'] = 0
ys['N_downshift_0min'] = 0
ys['Pheromone_0min'] = 0
ys['NaCl_0min'] = 0

m_fc = ys.unstack().reset_index()
m_fc.columns = ['condition', 'ion', 'fold-change']
m_fc['time'] = [int(i.split('_')[-1:][0].replace('min', '')) for i in m_fc['condition']]
m_fc['stimulation'] = [i.split('_')[0] if i.split('_')[0] in ['Pheromone', 'NaCl', 'Rapamycin'] else '_'.join(i.split('_')[:2]) for i in m_fc['condition']]
m_fc['unit'] = 0
m_fc['metabolite'] = [met_name[i] for i in m_fc['ion']]

features = ['L-Glutamine', 'L-Proline', 'N-Acetyl-L-glutamate', 'IMP']

plot_df = m_fc[[k in features for k in m_fc['metabolite']]].dropna()
plot_df = plot_df.drop([i for k in features for c in palette if len(plot_df[(plot_df['metabolite'] == k) & (plot_df['stimulation'] == c)]) <= 1 for i in plot_df[(plot_df['kinase'] == k) & (plot_df['stimulation'] == c)].index])

sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3}, font_scale=.75)
g = sns.FacetGrid(plot_df, col='metabolite', sharex=False, sharey=False, size=1.5, aspect=1, legend_out=True, col_order=features)
g.map_dataframe(sns.tsplot, time='time', unit='unit', condition='stimulation', value='fold-change', color=palette, marker='o', lw=.8)
g.map(plt.axhline, y=0, ls='-', lw=0.3, c='black', alpha=.5)
g.set_titles(col_template='{col_name}')
g.set_axis_labels('Time (minutes)', 'Fold-change')
g.add_legend()
plt.savefig('%s/reports/tsplot_m_foldchange.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Corr plotted!'


# -- Pairplot
for kinase, metabolite in it.product(['YFL033C', 'YJL141C', 'YJL164C', 'YJR066W', 'YBR160W'], ['L-Glutamine', 'L-Proline', 'N-Acetyl-L-glutamate', 'IMP', 'Pyruvate', 'Thiamine diphosphate']):
    m_ions = [k for k, v in met_name.items() if v == metabolite]
    print m_ions

    plot_df = DataFrame({'kinase': xs.ix[kinase] / xs.ix[kinase].std(), 'metabolite': ys.ix[m_ions[0]] - ys.ix[m_ions[0]].mean()})
    plot_df = plot_df[[not i.endswith('_0min') for i in plot_df.index]]

    color = '#808080'

    sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
    g = sns.jointplot(
        'metabolite', 'kinase', plot_df, 'reg', color=color, joint_kws={'scatter_kws': {'s': 40, 'edgecolor': 'w', 'linewidth': .5}},
        marginal_kws={'hist': False, 'rug': True}, space=0,
    )
    plt.axhline(0, ls='-', lw=0.3, c=color, alpha=.5)
    plt.axvline(0, ls='-', lw=0.3, c=color, alpha=.5)
    g.plot_marginals(sns.kdeplot, shade=True, color=color)
    g.set_axis_labels(metabolite, acc_name[kinase])
    plt.savefig('%s/reports/feature_correlation_%s_%s.pdf' % (wd, acc_name[kinase], metabolite), bbox_inches='tight')
    plt.close('all')
    print '[INFO] Corr plotted!'

