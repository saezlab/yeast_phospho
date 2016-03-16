import numpy as np
from yeast_phospho import wd
from pandas import DataFrame, read_csv
from pymist.enrichment.gsea import gsea
from yeast_phospho.utilities import get_tfs_targets


# Import growth rates
growth = read_csv('%s/files/strain_relative_growth_rate.txt' % wd, sep='\t', index_col=0)['relative_growth']
ko_strains = list(growth.index)

# Import TF targets
tf_targets = get_tfs_targets()
tf_targets = {t: set(tf_targets.ix[map(bool, tf_targets[t]), t].index) for t in tf_targets}

permuations = 10000


# -- Estimate TFs activities dynamic
dyn_trans = read_csv('%s/tables/transcriptomics_dynamic.tab' % wd, sep='\t', index_col=0)
dyn_trans = {c: dyn_trans[c].dropna().to_dict() for c in dyn_trans}

# Estimate TFs activities
tf_activity_dyn = {c: {t: gsea(dyn_trans[c], tf_targets[t], permuations) for t in tf_targets} for c in dyn_trans}
tf_activity_dyn = {c: {k: np.log10(tf_activity_dyn[c][k][1]) if tf_activity_dyn[c][k][0] > 0 else -np.log10(tf_activity_dyn[c][k][1]) for k in tf_activity_dyn[c]} for c in tf_activity_dyn}
tf_activity_dyn = DataFrame(tf_activity_dyn).dropna(how='all', axis=0)
tf_activity_dyn.to_csv('%s/tables/tf_activity_dynamic_gsea.tab' % wd, sep='\t')
print '[INFO] Activities estimated: dynamic'


# -- Estimate TFs activities steady-state
trans = read_csv('%s/tables/transcriptomics_steady_state.tab' % wd, sep='\t', index_col=0).loc[:, ko_strains].dropna(how='all', axis=1)
trans = {c: trans[c].dropna().to_dict() for c in trans}

# Estimate TFs activities
tf_activity = {c: {t: gsea(trans[c], tf_targets[t], permuations) for t in tf_targets} for c in trans}
tf_activity = {c: {k: np.log10(tf_activity[c][k][1]) if tf_activity[c][k][0] > 0 else -np.log10(tf_activity[c][k][1]) for k in tf_activity[c]} for c in tf_activity}
tf_activity = DataFrame(tf_activity).dropna(how='all', axis=0)
tf_activity.to_csv('%s/tables/tf_activity_steady_state_gsea.tab' % wd, sep='\t')
print '[INFO] Activities estimated: steady-state'
