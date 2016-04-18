import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas import DataFrame, read_csv
from yeast_phospho.utilities import get_kinases_targets, estimate_activity_with_sklearn, get_proteins_name


# Import growth rates
growth = read_csv('%s/files/strain_relative_growth_rate.txt' % wd, sep='\t', index_col=0)['relative_growth']
ko_strains = list(growth.index)

# Import kinase targets
k_targets = get_kinases_targets()


# -- Estimate kinase activities steady-state
phospho_df = read_csv('%s/tables/pproteomics_steady_state.tab' % wd, sep='\t', index_col=0).loc[:, ko_strains].dropna(how='all', axis=1)

# Estimate kinase activities
k_activity = DataFrame({c: estimate_activity_with_sklearn(k_targets, phospho_df[c].dropna()) for c in phospho_df})
k_activity.to_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t')


# -- Estimate kinase activities dynamic
# Import phospho FC
phospho_df_dyn = read_csv('%s/tables/pproteomics_dynamic.tab' % wd, sep='\t', index_col=0)

# Estimate kinase activities
k_activity_dyn = DataFrame({c: estimate_activity_with_sklearn(k_targets, phospho_df_dyn[c].dropna()) for c in phospho_df_dyn})
k_activity_dyn.to_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t')


# -- Estimate kinase activities of combination dynamic data
# Import uniprot mapping
acc = read_csv('%s/files/yeast_uniprot.txt' % wd, sep='\t', index_col=2)['oln'].to_dict()

# Import phospho FC
phospho_df_comb_dyn = read_csv('%s/tables/pproteomics_dynamic_combination.csv' % wd, index_col=0)
phospho_df_comb_dyn = phospho_df_comb_dyn[[i.split('_')[0] in acc for i in phospho_df_comb_dyn.index]]
phospho_df_comb_dyn.index = ['%s_%s' % (acc[i.split('_')[0]], i.split('_')[1]) for i in phospho_df_comb_dyn.index]

k_activity_comb_dyn = DataFrame({c: estimate_activity_with_sklearn(k_targets, phospho_df_comb_dyn[c].dropna()) for c in phospho_df_comb_dyn})
k_activity_comb_dyn.to_csv('%s/tables/kinase_activity_dynamic_combination.tab' % wd, sep='\t')
print '[INFO] Activities estimated'


# -- Number of measured targets
acc_name = get_proteins_name()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}

nitrogen_set, combination_set = set(phospho_df_dyn.index), set(phospho_df_comb_dyn.index)
total_set = nitrogen_set.union(combination_set)

k_targets_counts = DataFrame({k: [len(set(k_targets[k][k_targets[k] != 0].index).intersection(df)) for df in [nitrogen_set, combination_set, total_set]] for k in k_targets}, index=['nitrogen', 'combination', 'total']).T
k_targets_counts = k_targets_counts[(k_targets_counts['nitrogen'] > 1) & (k_targets_counts['combination'] > 1)]
k_targets_counts = k_targets_counts.sort('total')
k_targets_counts.index = [acc_name[i] for i in k_targets_counts.index]

sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3}, font_scale=.75)
k_targets_counts.plot(kind='barh', lw=0)
sns.despine()
plt.gcf().set_size_inches(2, 4)
plt.savefig('%s/reports/k_targets_counts_barplot.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Corr plotted!'
