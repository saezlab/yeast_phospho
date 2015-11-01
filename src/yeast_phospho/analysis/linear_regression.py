import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from sklearn.linear_model import Lasso
from sklearn.cross_validation import LeaveOneOut
from pandas import DataFrame, Series, read_csv
from yeast_phospho.utilities import pearson, get_proteins_name, get_metabolites_name, get_ko_strains

# KO strains
ko_strains = list(get_ko_strains())

# Import annotations
acc_name = get_proteins_name()
met_name = get_metabolites_name()


# ---- Import
# Steady-state with growth
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)
metabolomics = metabolomics[metabolomics.std(1) > .4]
metabolomics.index = [str(i) for i in metabolomics.index]
metabolomics = metabolomics[ko_strains]

k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)
k_activity = k_activity[(k_activity.count(1) / k_activity.shape[1]) > .75].replace(np.NaN, 0.0)
k_activity = k_activity[ko_strains]

tf_activity = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0)
tf_activity = tf_activity[ko_strains]


# Steady-state without growth
metabolomics_ng = read_csv('%s/tables/metabolomics_steady_state_no_growth.tab' % wd, sep='\t', index_col=0)
metabolomics_ng = metabolomics_ng[metabolomics_ng.std(1) > .4]

k_activity_ng = read_csv('%s/tables/kinase_activity_steady_state_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_ng = k_activity_ng[(k_activity_ng.count(1) / k_activity_ng.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_ng = read_csv('%s/tables/tf_activity_steady_state_no_growth.tab' % wd, sep='\t', index_col=0)


# Dynamic
metabolomics_dyn = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0)
metabolomics_dyn = metabolomics_dyn[metabolomics_dyn.std(1) > .4]
metabolomics_dyn.index = [str(i) for i in metabolomics_dyn.index]

k_activity_dyn = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0)
k_activity_dyn = k_activity_dyn[(k_activity_dyn.count(1) / k_activity_dyn.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_dyn = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0)


# ---- Build linear regression models
comparisons = [
    (k_activity, metabolomics, 'Kinases', 'Steady-state with growth'),
    (tf_activity, metabolomics, 'TFs', 'Steady-state with growth'),

    (k_activity_ng, metabolomics, 'Kinases', 'Steady-state without growth'),
    (tf_activity_ng, metabolomics, 'TFs', 'Steady-state without growth'),

    (k_activity_dyn, metabolomics_dyn, 'Kinases', 'Dynamic'),
    (tf_activity_dyn, metabolomics_dyn, 'TFs', 'Dynamic')
]


def loo_regressions(xs, ys, feature_type, dataset_type, lm=Lasso(alpha=0.01, max_iter=2000)):
    print '[INFO]', feature_type, dataset_type

    x = xs.loc[:, ys.columns].dropna(axis=1).T
    y = ys[x.index].T

    cv = LeaveOneOut(len(y))

    y_pred = DataFrame({x.index[test][0]: lm.fit(x.ix[train], y.ix[train]).predict(x.ix[test])[0] for train, test in cv}, index=y.columns)
    # y_coef = DataFrame({x.index[test][0]: lm.fit(x.ix[train], y.ix[train]).coef_ for train, test in cv}, index=y.columns)

    metabolites_corr = [(feature_type, dataset_type, f, 'metabolites', pearson(y.T.ix[f, y_pred.columns], y_pred.ix[f, y_pred.columns])[0]) for f in y_pred.index]
    conditions_corr = [(feature_type, dataset_type, s, 'conditions', pearson(y.T.ix[y_pred.index, s], y_pred.ix[y_pred.index, s])[0]) for s in y_pred]

    return metabolites_corr + conditions_corr

lm_res = [loo_regressions(xs, ys, ft, dt) for xs, ys, ft, dt in comparisons]
lm_res = [(ft, dt, f, ct, c) for c in lm_res for ft, dt, f, ct, c in c]
lm_res = DataFrame(lm_res, columns=['feature_type', 'dataset_type', 'variable', 'corr_type', 'cor'])
lm_res['variable_name'] = [acc_name[i] if i in acc_name else (met_name[i] if i in met_name else np.NaN) for i in lm_res['variable']]
print '[INFO] Regressions done'


sns.set(style='ticks')
g = sns.FacetGrid(lm_res, col='dataset_type', legend_out=True)
plt.ylim([-1, 1])
g.map(sns.boxplot, 'corr_type', 'cor', 'feature_type', palette='Set1', sym='')
g.map(sns.stripplot, 'corr_type', 'cor', 'feature_type', palette='Set1', jitter=True, size=5)
g.map(plt.axhline, y=0, ls='--', c='.5')
g.add_legend()
sns.despine(trim=True, bottom=True)
plt.savefig('%s/reports/lm_all.pdf' % wd, bbox_inches='tight')
plt.close('all')
