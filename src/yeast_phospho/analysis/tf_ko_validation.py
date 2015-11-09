import numpy as np
import seaborn as sns
import matplotlib
import itertools as it
import statsmodels.api as sm
import statsmodels.tools as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from yeast_phospho import wd, data
from sklearn.cross_validation import KFold, ShuffleSplit, LeaveOneOut
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics.scorer import make_scorer, mean_squared_error_scorer, mean_squared_error
from sklearn.decomposition.pca import PCA
from sklearn.feature_selection.rfe import RFECV, RFE
from sklearn.preprocessing.data import StandardScaler
from sklearn.linear_model import Lasso, RandomizedLasso, LinearRegression, LassoCV, ElasticNet, Ridge, RidgeCV
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression
from pandas import DataFrame, Series, read_csv, concat, melt, pivot_table
from yeast_phospho.utilities import pearson, get_proteins_name, get_metabolites_name, spearman


# Import annotations
acc_name = get_proteins_name()
met_name = get_metabolites_name()


# Import data-set
tf_activity = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0)

k_activity = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0)
k_activity = k_activity[(k_activity.count(1) / k_activity.shape[1]) > .75].replace(np.NaN, 0.0)

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

    return df

tf_logfc = import_tf_ko('%s/yeast_tf/yeast_tf_ko_logfc.txt' % data)
tf_pvalue = import_tf_ko('%s/yeast_tf/yeast_tf_ko_pvalues.txt' % data)


# Linear regression
sc_x, sc_y = StandardScaler(), StandardScaler()

x = tf_activity.T
x = DataFrame(sc_x.fit_transform(x), index=x.index, columns=x.columns)

y = metabolomics[x.index].T
y = DataFrame(sc_y.fit_transform(y), index=y.index, columns=y.columns)

fits = {}
for m in y.columns:
    xs = x.copy()
    ys = y.ix[x.index, m]

    cv = ShuffleSplit(len(ys), n_iter=30, test_size=.2)
    lm = Lasso(alpha=1)

    rfe = RFECV(lm, 1, cv=cv, scoring='mean_squared_error').fit(xs, ys)
    print '[INFO] %s: %d' % (m, rfe.n_features_)

    lm = lm.fit(xs.ix[:, rfe.get_support()], ys)

    xs = xs.ix[:, rfe.get_support()]

    y_pred = Series({ys.ix[test].index[0]: lm.fit(xs.ix[train], ys.ix[train]).predict(xs.ix[test])[0] for train, test in LeaveOneOut(len(y[m]))})

    cor, pvalue, _ = pearson(y_pred[ys.index].values, ys.values)

    fits[m] = (lm, rfe, cor, pvalue)

print '[INFO] Regressions done: ', len(fits)


sns.set(style='ticks')
gs, pos = GridSpec(len(y.columns), 3, hspace=.5), 0
matplotlib.pyplot.gcf().set_size_inches(12, 3 * len(y.columns))

for m in y.columns:
    lm, rfe, cor, pvalue = fits[m]

    ax = plt.subplot(gs[pos])
    sns.residplot(y[m], lm.predict(x.ix[y.index, rfe.get_support()]) - y[m], lowess=True, ax=ax)
    ax.set_title('Residual plot - %s' % m)
    ax.set_xlabel('meas')
    ax.set_ylabel('pred - meas')
    sns.despine(trim=True, ax=ax)

    ax = plt.subplot(gs[pos + 1])
    ax.plot(range(1, len(rfe.grid_scores_) + 1), (-rfe.grid_scores_), c='#95a5a6', lw=.8)
    ax.axvline(rfe.n_features_, c='#95a5a6', lw=.3)
    ax.set_title('Optimal: %d' % rfe.n_features_)
    ax.set_xlabel('# features')
    ax.set_ylabel('rmse')
    sns.despine(trim=True, ax=ax)

    res = []
    for train, test in ShuffleSplit(len(y[m]), n_iter=30, test_size=.2):
        lm_m = lm.fit(x.ix[train, rfe.get_support()], y.ix[train, m])
        res.append((rmse(lm_m.predict(x.ix[train, rfe.get_support()]), y.ix[train, m]), 'train'))
        res.append((rmse(lm_m.predict(x.ix[test, rfe.get_support()]), y.ix[test, m]), 'test'))

    res = DataFrame(res, columns=['rmse', 'type'])

    ax = plt.subplot(gs[pos + 2])
    sns.boxplot('type', 'rmse', 'type', res, sym='', ax=ax)
    sns.stripplot('type', 'rmse', 'type', res, marker='o', size=8, jitter=True, ax=ax)
    ax.set_title('Test/Train RMSEs')
    sns.despine(trim=True, ax=ax)
    ax.legend().remove()

    pos += 3

plt.savefig('%s/reports/TF_KO_validation_fitting_tests.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Regressions plot done'


df = [(m, f, c) for m in fits for f, c in zip(*(x.ix[:, fits[m][1].get_support()], fits[m][0].coef_)) if c != 0 and fits[m][2] > 0 and fits[m][3] < .05]
df = DataFrame(df, columns=['variable', 'feature', 'coef'])
df['cor'] = [spearman(y[m].values, x[f].values)[0] for m, f in df[['variable', 'feature']].values]
# df['ci_lb'] = [x_coefs[m][f].conf_int().ix[f, 0] for m, f in df[['variable', 'feature']].values]
# df['ci_ub'] = [x_coefs[m][f].conf_int().ix[f, 1] for m, f in df[['variable', 'feature']].values]
df['var_name'] = [met_name[i] for i in df['variable']]
df['feature_name'] = [acc_name[i] for i in df['feature']]
df.to_csv('%s/tables/tf_coefficients.txt' % wd, sep='\t', index=False)
print df.head(15)


plot_df = [(m, f, coef, tf_logfc.ix[m, c]) for m, f, coef in df[['variable', 'feature', 'coef']].values if m in tf_logfc.index for c in tf_logfc if f in c]
plot_df = DataFrame(plot_df, columns=['variable', 'feature', 'coef', 'fold-change'])

sns.set(style='ticks')
margin = 1.20
xlim, ylim = (plot_df['coef'].min() * margin, plot_df['coef'].max() * margin), (plot_df['fold-change'].min() * margin, plot_df['fold-change'].max() * margin)
sns.jointplot('coef', 'fold-change', data=plot_df, kind='reg', xlim=xlim, ylim=ylim, marginal_kws={'hist': False})
plt.axhline(y=0, ls='--', c='.5', lw=.3)
plt.axvline(x=0, ls='--', c='.5', lw=.3)
plt.xlabel('TF (coefficient)')
plt.ylabel('TF knockout (log fold-change)')
plt.savefig('%s/reports/TF_KO_validation.pdf' % wd, bbox_inches='tight')
plt.close('all')
