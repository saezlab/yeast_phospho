from yeast_phospho import wd
from pandas import DataFrame, read_csv
from yeast_phospho.utilities import get_kinases_targets, estimate_activity_with_gsea


# Import growth rates
growth = read_csv('%s/files/strain_relative_growth_rate.txt' % wd, sep='\t', index_col=0)['relative_growth']
ko_strains = list(growth.index)

# Import kinase targets
k_targets = get_kinases_targets()

permuations = 10000

# ---- Estimate kinase activities steady-state
phospho_df = read_csv('%s/tables/pproteomics_steady_state.tab' % wd, sep='\t', index_col=0).loc[:, ko_strains].dropna(how='all', axis=1)

# Estimate kinase activities
k_activity = DataFrame({c: {t: estimate_activity_with_gsea(phospho_df[c].dropna().to_dict(), set(k_targets.loc[k_targets[t] != 0, t].index), permuations) for t in k_targets} for c in phospho_df})
k_activity.to_csv('%s/tables/kinase_activity_steady_state_gsea.tab' % wd, sep='\t')


# ---- Estimate kinase activities dynamic
# Import phospho FC
phospho_df_dyn = read_csv('%s/tables/pproteomics_dynamic.tab' % wd, sep='\t', index_col=0)

# Estimate kinase activities
k_activity_dyn = DataFrame({c: {t: estimate_activity_with_gsea(phospho_df_dyn[c].dropna().to_dict(), set(k_targets.loc[k_targets[t] != 0, t].index), permuations) for t in k_targets} for c in phospho_df_dyn})
k_activity_dyn.to_csv('%s/tables/kinase_activity_dynamic_gsea.tab' % wd, sep='\t')

print '[INFO] Activities estimated'
