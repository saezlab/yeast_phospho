import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from yeast_phospho.utilities import pearson
from pandas import DataFrame, Series, read_csv
from yeast_phospho.utilities import get_tfs_targets, estimate_activity_with_sklearn, regress_out


# Import growth rates
growth = read_csv('%s/files/strain_relative_growth_rate.txt' % wd, sep='\t', index_col=0)['relative_growth']

# Import TF targets
tf_targets = get_tfs_targets()


# ---- Estimate TFs activities steady-state
trans = read_csv('%s/tables/transcriptomics_steady_state.tab' % wd, sep='\t', index_col=0).dropna()

# Filter gene-expression to overlapping conditions
trans = trans.loc[:, growth.index].dropna(axis=1)

# Estimate TFs activities
tf_activity = DataFrame({c: estimate_activity_with_sklearn(tf_targets, trans[c]) for c in trans})
tf_activity.to_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t')

# Regress-out growth
tf_activity = DataFrame({k: regress_out(growth[tf_activity.columns], tf_activity.ix[k]) for k in tf_activity.index}).T
tf_activity.to_csv('%s/tables/tf_activity_steady_state_no_growth.tab' % wd, sep='\t')


# ---- Estimate TFs activities dynamic
dyn_trans_df = read_csv('%s/tables/transcriptomics_dynamic.tab' % wd, sep='\t', index_col=0)

# Estimate TFs activities
tf_activity_dyn = DataFrame({c: estimate_activity_with_sklearn(tf_targets, dyn_trans_df[c]) for c in dyn_trans_df})
tf_activity_dyn.to_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t')
