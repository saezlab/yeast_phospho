import numpy as np
from pandas.stats.misc import zscore
from yeast_phospho import wd
from sklearn.linear_model import Ridge, LinearRegression
from pandas import DataFrame, read_csv

ridge = Ridge(alpha=.1)

# Import growth rates
growth = read_csv('%s/files/strain_relative_growth_rate.txt' % wd, sep='\t', index_col=0)['relative_growth']

# Import TF targets
tf_targets = read_csv('%s/tables/targets_tfs.tab' % wd, sep='\t', index_col=0)

# Import steady-state transcriptomics
trans = read_csv('%s/tables/transcriptomics_steady_state.tab' % wd, sep='\t', index_col=0).dropna()

# Overlap conditions with growth measurements
strains = list(set(trans.columns).intersection(growth.index))

# Filter gene-expression to overlapping conditions
trans = trans[strains]


def calculate_activity(strain):
    y = trans[strain].dropna()
    x = tf_targets.ix[y.index].replace(np.NaN, 0.0)

    x = x.loc[:, x.sum() != 0]

    return dict(zip(*(x.columns, ridge.fit(x, zscore(y)).coef_)))

tf_activity = DataFrame({c: calculate_activity(c) for c in strains})
print '[INFO] TF activity calculated: ', tf_activity.shape

tf_activity.to_csv('%s/tables/tf_activity_steady_state_with_growth.tab' % wd, sep='\t')
print '[INFO] [TF ACTIVITY] Exported '

# Regress out growth


def regress_out_growth(kinase):
    x, y = growth.ix[strains].values, tf_activity.ix[kinase, strains].values

    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))

    x, y = x[mask], y[mask]

    lm = LinearRegression().fit(np.mat(x).T, y)

    y_ = y - lm.coef_[0] * x - lm.intercept_

    return dict(zip(np.array(strains)[mask], y_))

tf_activity = DataFrame({kinase: regress_out_growth(kinase) for kinase in tf_activity.index}).T.dropna(axis=0, how='all')
print '[INFO] Growth regressed out from the Kinases activity scores: ', tf_activity.shape

# Export kinase activity matrix
tf_activity.to_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t')
print '[INFO] [TF ACTIVITY] Exported'


# ---- Dynamic: gene-expression data-set
dyn_trans_df = read_csv('%s/tables/transcriptomics_dynamic.tab' % wd, sep='\t', index_col=0)

conditions = list(dyn_trans_df.columns)


def calculate_activity(condition):
    y = dyn_trans_df[condition].dropna()
    x = tf_targets.ix[y.index].replace(np.NaN, 0)

    x = x.loc[:, x.sum() != 0]

    return dict(zip(*(x.columns, ridge.fit(x, zscore(y)).coef_)))

tf_activity = DataFrame({c: calculate_activity(c) for c in conditions})
print '[INFO] TF activity calculated: ', tf_activity.shape

tf_activity.to_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t')
print '[INFO] [TF ACTIVITY] Exported '
