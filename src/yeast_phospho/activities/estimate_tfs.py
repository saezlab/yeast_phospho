from yeast_phospho import wd
from pandas import DataFrame, read_csv
from yeast_phospho.utilities import get_tfs_targets, estimate_activity_with_sklearn


# Import growth rates
growth = read_csv('%s/files/strain_relative_growth_rate.txt' % wd, sep='\t', index_col=0)['relative_growth']
ko_strains = list(growth.index)

# Import TF targets
tf_targets = get_tfs_targets()


# -- Estimate TFs activities steady-state
trans = read_csv('%s/tables/transcriptomics_steady_state.tab' % wd, sep='\t', index_col=0).loc[:, ko_strains].dropna(how='all', axis=1)

# Estimate TFs activities
tf_activity = DataFrame({c: estimate_activity_with_sklearn(tf_targets, trans[c]) for c in trans})
tf_activity.to_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t')


# -- Estimate TFs activities dynamic
dyn_trans_df = read_csv('%s/tables/transcriptomics_dynamic.tab' % wd, sep='\t', index_col=0)

# Estimate TFs activities
tf_activity_dyn = DataFrame({c: estimate_activity_with_sklearn(tf_targets, dyn_trans_df[c]) for c in dyn_trans_df})
tf_activity_dyn.to_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t')
