import numpy as np
import seaborn as sns
import itertools as it
import statsmodels.api as sm
import statsmodels.tools as st
import matplotlib.pyplot as plt
from yeast_phospho import wd, data
from sklearn.cross_validation import KFold, ShuffleSplit
from statsmodels.tools.eval_measures import rmse
from scipy.stats.stats import spearmanr, pearsonr
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression
from pandas import DataFrame, Series, read_csv, concat, melt, pivot_table
from yeast_phospho.utilities import pearson, get_proteins_name, get_metabolites_name


# Import annotations
acc_name = get_proteins_name()
met_name = get_metabolites_name()


# Import data-set
tf_activity = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0)

metabolomics = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0)[tf_activity.columns]
metabolomics = metabolomics[metabolomics.std(1) > .4]
metabolomics.index = [str(i) for i in metabolomics.index]
metabolomics = metabolomics[[i in met_name for i in metabolomics.index]]


# Import TF KO data-set metabolomics
tf_ko_map = read_csv('%s/yeast_tf/tf_map.txt' % data, sep='\t', index_col=0)['orf'].to_dict()


def import_tf_ko(file_path):
    df = read_csv(file_path, sep='\t', index_col=0)
    df.index = ['%.2f' % i for i in df.index]

    counts = {mz: counts for mz, counts in zip(*(np.unique(df.index, return_counts=True)))}
    df = df[[counts[i] == 1 for i in df.index]]

    df.columns = ['_'.join(c.split('_')[:-1]) for c in df]
    df = df[list(set(df).intersection(tf_ko_map))]
    df.columns = [c.split('_')[0] + '_' + tf_ko_map[c] for c in df.columns]

    df = df.ix[metabolomics.index].dropna()

    df = df.loc[:, [c.split('_')[1] in tf_activity.index for c in df]]

    return df

tf_logfc = import_tf_ko('%s/yeast_tf/yeast_tf_ko_logfc.txt' % data)
tf_pvalue = import_tf_ko('%s/yeast_tf/yeast_tf_ko_pvalues.txt' % data)


# Linear regression
y = metabolomics.T
x = tf_activity[y.index].T

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

df = [(f, m, x_coefs[m].params[f], x_coefs[m].pvalues[f], x_coefs[m].conf_int().ix[f, 0], x_coefs[m].conf_int().ix[f, 1]) for m in x_coefs if m in tf_logfc.index for f, p in x_coefs[m].pvalues.to_dict().items() if f != 'const']
df = [(tf, m, c, v, p, tf_logfc.ix[m, c], tf_pvalue.ix[m, c], cilb, ciub) for tf, m, v, p, cilb, ciub in df for c in [c for c in tf_logfc if tf in c]]
df = DataFrame(df, columns=['TF', 'Metabolite', 'Condition', 'coef', 'coef_pvalue', 'fold-change', 'pvalue', 'ci_lb', 'ci_ub']).dropna()
df['Metabolite_name'] = [met_name[i] for i in df['Metabolite']]
df['TF_name'] = [acc_name[i] for i in df['TF']]
print df.sort('pvalue')
df.sort(['coef_pvalue', 'pvalue']).to_csv('%s/tables/tf_coefficients.txt' % wd, sep='\t', index=False)

sns.set(style='ticks')
margin = 1.20
xlim, ylim = (df['coef'].min() * margin, df['coef'].max() * margin), (df['fold-change'].min() * margin, df['fold-change'].max() * margin)
sns.jointplot('coef', 'fold-change', data=df, kind='reg', xlim=xlim, ylim=ylim, marginal_kws={'hist': False}, stat_func=pearsonr)
plt.axhline(y=0, ls='--', c='.5', lw=.3)
plt.axvline(x=0, ls='--', c='.5', lw=.3)
plt.xlabel('TF (coefficient)')
plt.ylabel('TF knockout (log fold-change)')
plt.savefig('%s/reports/TF_KO_validation.pdf' % wd, bbox_inches='tight')
plt.close('all')
