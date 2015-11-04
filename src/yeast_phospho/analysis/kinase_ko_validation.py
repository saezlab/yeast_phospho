import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tools as st
from yeast_phospho import wd, data
from scipy.stats.stats import spearmanr
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression
from pandas import DataFrame, Series, read_csv, concat, melt, pivot_table
from yeast_phospho.utilities import pearson, get_proteins_name, get_metabolites_name


# Import annotations
acc_name = get_proteins_name()
met_name = get_metabolites_name()


# Import data-set
metabolomics = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0)
metabolomics = metabolomics[metabolomics.std(1) > .4]
metabolomics.index = [str(i) for i in metabolomics.index]

k_activity = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0)
k_activity = k_activity[(k_activity.count(1) / k_activity.shape[1]) > .75].replace(np.NaN, 0.0)


# Linear regression
y = metabolomics.T
x = k_activity[y.index].T

x_coefs = {}
for m in y:
    fs = SelectKBest(f_regression, k=10).fit(x, y[m]).get_support()

    xs = x.loc[:, fs]
    xs = st.add_constant(xs)

    lm = sm.OLS(y[m], xs).fit_regularized(alpha=.01, L1_wt=.5)

    x_coefs[m] = lm

x_coefs = {m: x_coefs[m] for m in x_coefs if x_coefs[m].rsquared_adj > .5 and x_coefs[m].f_pvalue < .05}
print '[INFO] Regressions done: ', len(x_coefs)


# Interactions table
df = [(f, m, x_coefs[m].params[f], x_coefs[m].pvalues[f], x_coefs[m].conf_int().ix[f, 0], x_coefs[m].conf_int().ix[f, 1]) for m in x_coefs for f, p in x_coefs[m].pvalues.to_dict().items() if f != 'const' and m in met_name]
df = DataFrame(df, columns=['kinase', 'metabolite', 'coef', 'pvalue', 'ci_lb', 'ci_ub']).dropna().sort('pvalue')
df['metabolite_name'] = [met_name[i] for i in df['metabolite']]
df['kinase_name'] = [acc_name[i] for i in df['kinase']]
print df[df['pvalue'] < .05].sort('pvalue')
df.sort('pvalue').to_csv('%s/tables/kinases_coefficients.txt' % wd, sep='\t', index=False)
