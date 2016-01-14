import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas import DataFrame, read_csv
from yeast_phospho.utilities import estimate_activity_with_sklearn, get_kinases_targets, estimate_activity_with_statsmodel


# Import growth rates
growth = read_csv('%s/files/strain_relative_growth_rate.txt' % wd, sep='\t', index_col=0)['relative_growth']
ko_strains = list(growth.index)

# Import kinase targets
k_targets = get_kinases_targets()


# ---- Estimate kinase activities steady-state
phospho_df = read_csv('%s/tables/pproteomics_steady_state.tab' % wd, sep='\t', index_col=0).loc[:, ko_strains].dropna(how='all', axis=1)

# Estimate kinase activities
k_activity = DataFrame({c: estimate_activity_with_statsmodel(k_targets, phospho_df[c], alpha=.01) for c in phospho_df})
k_activity.to_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t')


# ---- Estimate kinase activities dynamic
# Import phospho FC
phospho_df_dyn = read_csv('%s/tables/pproteomics_dynamic.tab' % wd, sep='\t', index_col=0)

# Estimate kinase activities
k_activity_dyn = DataFrame({c: estimate_activity_with_statsmodel(k_targets, phospho_df_dyn[c], alpha=.01) for c in phospho_df_dyn})
k_activity_dyn.to_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t')
print '[INFO] Activities estimated'
