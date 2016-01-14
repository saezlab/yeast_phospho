import numpy as np
from yeast_phospho import wd
from pandas import DataFrame, read_csv
from pymist.enrichment.gsea import gsea
from yeast_phospho.utilities import get_kinases_targets

# Import kinase targets
k_targets = get_kinases_targets()
k_targets = {t: set(k_targets.ix[map(bool, k_targets[t]), t].index) for t in k_targets}

permuations = 10000


# -- Estimate kinase activities steady-state
# Import growth rates
growth = read_csv('%s/files/strain_relative_growth_rate.txt' % wd, sep='\t', index_col=0)['relative_growth']
ko_strains = list(growth.index)

# Import phospho FC
phospho_df = read_csv('%s/tables/pproteomics_steady_state.tab' % wd, sep='\t', index_col=0).loc[:, ko_strains].dropna(how='all', axis=1)
phospho_df = {c: phospho_df[c].dropna().to_dict() for c in phospho_df}

# Estimate kinase activities
k_activity = {c: {k: gsea(phospho_df[c], k_targets[k], permuations) for k in k_targets} for c in phospho_df}
k_activity = {c: {k: np.log10(k_activity[c][k][1]) if k_activity[c][k][0] > 0 else -np.log10(k_activity[c][k][1]) for k in k_activity[c]} for c in k_activity}
k_activity = DataFrame(k_activity).dropna(how='all', axis=0)
k_activity.to_csv('%s/tables/kinase_activity_steady_state_gsea.tab' % wd, sep='\t')


# -- Estimate kinase activities dynamic
# Import phospho FC
phospho_df_dyn = read_csv('%s/tables/pproteomics_dynamic.tab' % wd, sep='\t', index_col=0)
phospho_df_dyn = {c: phospho_df_dyn[c].dropna().to_dict() for c in phospho_df_dyn}

# Estimate kinase activities
k_activity_dyn = {c: {t: gsea(phospho_df_dyn[c], k_targets[t], permuations) for t in k_targets} for c in phospho_df_dyn}
k_activity_dyn = {c: {k: np.log10(k_activity_dyn[c][k][1]) if k_activity_dyn[c][k][0] > 0 else -np.log10(k_activity_dyn[c][k][1]) for k in k_activity_dyn[c]} for c in k_activity_dyn}
k_activity_dyn = DataFrame(k_activity_dyn).dropna(how='all', axis=0)
k_activity_dyn.to_csv('%s/tables/kinase_activity_dynamic_gsea.tab' % wd, sep='\t')


# -- Estimate kinase activities of combination dynamic data
# Import uniprot mapping
acc = read_csv('%s/files/yeast_uniprot.txt' % wd, sep='\t', index_col=2)['oln'].to_dict()

# Import phospho FC
phospho_df_comb_dyn = read_csv('%s/tables/pproteomics_dynamic_combination.csv' % wd, index_col=0)
phospho_df_comb_dyn = phospho_df_comb_dyn[[i.split('_')[0] in acc for i in phospho_df_comb_dyn.index]]
phospho_df_comb_dyn.index = ['%s_%s' % (acc[i.split('_')[0]], i.split('_')[1]) for i in phospho_df_comb_dyn.index]
phospho_df_comb_dyn = {c: phospho_df_comb_dyn[c].dropna().to_dict() for c in phospho_df_comb_dyn}

k_activity_comb_dyn = {c: {k: gsea(phospho_df_comb_dyn[c], k_targets[k], permuations) for k in k_targets} for c in phospho_df_comb_dyn}
k_activity_comb_dyn = {c: {k: np.log10(k_activity_comb_dyn[c][k][1]) if k_activity_comb_dyn[c][k][0] > 0 else -np.log10(k_activity_comb_dyn[c][k][1]) for k in k_activity_comb_dyn[c]} for c in k_activity_comb_dyn}
k_activity_comb_dyn = DataFrame(k_activity_comb_dyn).dropna(how='all', axis=0)
k_activity_comb_dyn.to_csv('%s/tables/kinase_activity_dynamic_combination_gsea.tab' % wd, sep='\t')
print '[INFO] Activities estimated'
