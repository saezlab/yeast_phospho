import numpy as np
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tools as st
from yeast_phospho import wd, data
from scipy.stats.stats import spearmanr, pearsonr
from sklearn.cross_validation import KFold, ShuffleSplit
from statsmodels.tools.eval_measures import rmse
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

L1_wt_list = [0, .5, 1.]
alpha_list = [.1, 1., 10.]
nfeat_list = [1, 5, 10]

x_coefs = {}
for m in y:
    # Train parameters
    res = (np.Inf, np.Inf, np.Inf, np.Inf)
    for L1_wt, alpha, n_feat in it.product(L1_wt_list, alpha_list, nfeat_list):
        fs = SelectKBest(f_regression, k=n_feat).fit(x, y[m]).get_support()

        xs = x.loc[:, fs]
        xs = st.add_constant(xs)

        cv = ShuffleSplit(len(y[m]), n_iter=30, test_size=.15)

        p_rmse = np.median([rmse(y.ix[test, m], sm.OLS(y.ix[train, m], xs.ix[train]).fit_regularized(alpha=alpha, L1_wt=L1_wt).predict(xs.ix[test])) for train, test in cv])

        if res[0] > p_rmse:
            res = (p_rmse, L1_wt, alpha, n_feat)

    p_rmse, L1_wt, alpha, n_feat = res

    # Fit model
    fs = SelectKBest(f_regression, k=n_feat).fit(x, y[m]).get_support()

    xs = x.loc[:, fs]
    xs = st.add_constant(xs)

    lm = sm.OLS(y[m], xs).fit_regularized(alpha=alpha, L1_wt=L1_wt)

    x_coefs[m] = lm

    print m, p_rmse, L1_wt, alpha, n_feat, lm.rsquared, lm.f_pvalue

x_coefs = {m: x_coefs[m] for m in x_coefs if x_coefs[m].f_pvalue < 0.05}
print '[INFO] Regressions done: ', len(x_coefs)


# Interactions table
df = [(f, m, x_coefs[m].rsquared_adj, x_coefs[m].params[f], x_coefs[m].pvalues[f], x_coefs[m].conf_int().ix[f, 0], x_coefs[m].conf_int().ix[f, 1]) for m in x_coefs for f, p in x_coefs[m].pvalues.to_dict().items() if f != 'const' and m in met_name]
df = DataFrame(df, columns=['kinase', 'metabolite', 'rsquared_adj', 'coef', 'pvalue', 'ci_lb', 'ci_ub']).dropna().sort('pvalue')
df['metabolite_name'] = [met_name[i] for i in df['metabolite']]
df['kinase_name'] = [acc_name[i] for i in df['kinase']]
print df[df['pvalue'] < .1].sort('rsquared_adj', ascending=False)
df.sort('pvalue').sort('rsquared_adj', ascending=False).to_csv('%s/tables/kinases_coefficients.txt' % wd, sep='\t', index=False)


#
metabolomics_ss = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)
metabolomics_ss.index = [str(i) for i in metabolomics_ss.index]


#
plot_df = df[[k in metabolomics_ss and m in metabolomics_ss.index for k, m in df[['kinase', 'metabolite']].values]]
plot_df['fold-change'] = [metabolomics_ss.ix[m, k] for k, m in plot_df[['kinase', 'metabolite']].values]

sns.set(style='ticks')
margin = 1.20
xlim, ylim = (plot_df['coef'].min() * margin, plot_df['coef'].max() * margin), (plot_df['fold-change'].min() * margin, plot_df['fold-change'].max() * margin)
sns.jointplot('coef', 'fold-change', data=plot_df, kind='reg', xlim=xlim, ylim=ylim, marginal_kws={'hist': False}, stat_func=pearsonr)
plt.axhline(y=0, ls='--', c='.5', lw=.3)
plt.axvline(x=0, ls='--', c='.5', lw=.3)
plt.xlabel('Kinase (coefficient)')
plt.ylabel('Kinase knockout (log fold-change)')
plt.savefig('%s/reports/Kinase_KO_validation.pdf' % wd, bbox_inches='tight')
plt.close('all')
