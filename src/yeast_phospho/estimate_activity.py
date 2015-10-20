import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tools as st
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas.stats.misc import zscore
from sklearn.linear_model import Ridge
from pandas import DataFrame, Series, read_csv, pivot_table


def get_kinases_targets(studies_to_filter={'21177495'}):
    k_targets = read_csv('%s/files/PhosphoGrid.txt' % wd, sep='\t')

    k_targets = k_targets[[len(set(i.split('|')).intersection(studies_to_filter)) != len(set(i.split('|'))) for i in k_targets['KINASES_EVIDENCE_PUBMED']]]
    k_targets = k_targets[[len(set(i.split('|')).intersection(studies_to_filter)) != len(set(i.split('|'))) for i in k_targets['PHOSPHATASES_EVIDENCE_PUBMED']]]

    k_targets = k_targets.loc[(k_targets['KINASES_ORFS'] != '-') | (k_targets['PHOSPHATASES_ORFS'] != '-')]

    k_targets['SOURCE'] = k_targets['KINASES_ORFS'] + '|' + k_targets['PHOSPHATASES_ORFS']
    k_targets = [(k, t + '_' + site) for t, site, source in k_targets[['ORF_NAME', 'PHOSPHO_SITE', 'SOURCE']].values for k in source.split('|') if k != '-' and k != '']
    k_targets = DataFrame(k_targets, columns=['kinase', 'site'])

    k_targets['value'] = 1

    k_targets = pivot_table(k_targets, values='value', index='site', columns='kinase', fill_value=0)

    return k_targets


def kinase_activity_with_sklearn(x, y, alpha=.1):
    ys = y.dropna()
    xs = x.ix[ys.index].replace(np.NaN, 0.0)

    xs = xs.loc[:, xs.sum() != 0]

    lm = Ridge(fit_intercept=True, alpha=alpha).fit(xs, zscore(ys))

    return dict(zip(*(xs.columns, lm.coef_)))


def kinase_activity_with_statsmodel(x, y, alpha=.1):
    ys = y.dropna()
    xs = x.ix[ys.index].replace(np.NaN, 0.0)

    xs = xs.loc[:, xs.sum() != 0]

    lm = sm.OLS(zscore(ys), st.add_constant(xs))

    res = lm.fit_regularized(L1_wt=0, alpha=alpha)

    return res.params.drop('const').to_dict()
