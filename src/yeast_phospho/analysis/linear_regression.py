import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tools as st
from yeast_phospho import wd
from sklearn.linear_model import Lasso
from sklearn.cross_validation import LeaveOneOut
from pandas import DataFrame, read_csv
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression
from yeast_phospho.utilities import pearson, get_proteins_name, get_metabolites_name


# Import annotations
acc_name = get_proteins_name()
met_name = get_metabolites_name()


# ---- Import
# Steady-state with growth
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)
metabolomics = metabolomics[metabolomics.std(1) > .4]
metabolomics.index = [str(i) for i in metabolomics.index]

k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)
k_activity = k_activity[(k_activity.count(1) / k_activity.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0)


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
    (k_activity, metabolomics, 'Kinases', 'Steady-state with growth', 15),
    (tf_activity, metabolomics, 'TFs', 'Steady-state with growth', 15),

    (k_activity_ng, metabolomics, 'Kinases', 'Steady-state without growth', 15),
    (tf_activity_ng, metabolomics, 'TFs', 'Steady-state without growth', 15),

    (k_activity_dyn, metabolomics_dyn, 'Kinases', 'Dynamic', 10),
    (tf_activity_dyn, metabolomics_dyn, 'TFs', 'Dynamic', 10)
]


def loo_regressions(xs, ys, feature_type, dataset_type, k=15):
    print '[INFO]', feature_type, dataset_type

    x = xs.loc[:, ys.columns].dropna(axis=1).T
    y = ys[x.index].T

    cv = LeaveOneOut(len(y))

    y_pred = {}
    for m in y:
        y_pred[m] = {}
        for train, test in cv:
            best_features = SelectKBest(f_regression, k=k).fit(x.ix[train], y.ix[train, m]).get_support()

            # lm=Lasso(alpha=0.01, max_iter=2000)
            # y_pred[m][x.index[test][0]] = lm.fit(x.ix[train, best_features], y.ix[train, m]).predict(x.ix[test, best_features])[0]

            lm = sm.OLS(y.ix[train, m], st.add_constant(x.ix[train, best_features])).fit_regularized(alpha=.01, L1_wt=0.5)
            y_pred[m][x.index[test][0]] = lm.predict(st.add_constant(x.ix[test, best_features]))[0]

    y_pred = DataFrame(y_pred).T
    print '[INFO] Regression done: ', feature_type, dataset_type

    metabolites_corr = [(feature_type, dataset_type, f, 'metabolites', pearson(y.T.ix[f, y_pred.columns], y_pred.ix[f, y_pred.columns])[0]) for f in y_pred.index]
    conditions_corr = [(feature_type, dataset_type, s, 'conditions', pearson(y.T.ix[y_pred.index, s], y_pred.ix[y_pred.index, s])[0]) for s in y_pred]

    return metabolites_corr + conditions_corr

lm_res = [loo_regressions(xs, ys, ft, dt, fs) for xs, ys, ft, dt, fs in comparisons]
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
