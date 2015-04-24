import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.stats.misc import zscore
from scipy.stats.stats import spearmanr, pearsonr
from pandas import DataFrame, read_csv, Index, pivot_table, melt
from sklearn.cross_validation import KFold, LeaveOneOut
from sklearn.feature_selection.univariate_selection import f_classif, SelectKBest, chi2, f_regression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics.metrics import r2_score
from sklearn.preprocessing.imputation import Imputer


wd = '/Users/emanuel/Projects/projects/yeast_phospho/'

# Version
sns.set_style('white')

version = 'v10'
print '[INFO] Version: %s' % version

# Import phospho FC
phospho_df = read_csv(wd + 'tables/steady_state_phosphoproteomics.tab', sep='\t', index_col='site')
dyn_phospho_fc = read_csv(wd + 'tables/dynamic_phosphoproteomics.tab', sep='\t', index_col='site')

# Import metabolites map
m_map = read_csv(wd + 'files/metabolomics/metabolite_mz_map_kegg.txt', sep='\t')  # _adducts
m_map['mz'] = ['%.2f' % c for c in m_map['mz']]

# Import steady-state data-sets kinase enrichment
kinase_df = read_csv(wd + 'tables/kinase_enrichment_df.tab', sep='\t', index_col=0)

metabol_df = read_csv(wd + 'tables/steady_state_metabolomics.tab', sep='\t', index_col=0)
metabol_df.index = Index(['%.2f' % c for c in metabol_df.index], dtype=str)

# Import dynamic data-sets
dyn_kinase_df = read_csv(wd + 'tables/kinase_enrichment_dynamic_df.tab', sep='\t', index_col=0)

dyn_metabol_df = read_csv(wd + 'tables/dynamic_metabolomics.tab', sep='\t', index_col=0)
dyn_metabol_df.index = Index(dyn_metabol_df.index, dtype=str)

# Import growth rates
growth = read_csv(wd + 'files/strain_relative_growth_rate.txt', sep='\t', index_col=0)

# Import acc map to name form uniprot
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)

# Import kinases targets
kinases_targets = read_csv(wd + 'tables/kinases_phosphatases_targets.tab', sep='\t')
kinases = set(kinases_targets['SOURCE'])
kinases_targets = {k: set(kinases_targets.loc[kinases_targets['SOURCE'] == k, 'TARGET']) for k in kinases}

# Overlapping kinases/phosphatases knockout
strains = list(set(kinase_df.columns).intersection(set(metabol_df.columns)))
kinase_df, metabol_df = kinase_df[strains], metabol_df[strains]

# Machine learning: metabolites -> growth
X, Y = kinase_df.copy(), metabol_df.copy()

Xs, Ys = Y.copy().T.values, growth.loc[strains, 'relative_growth'].copy().values / 100

scores = []
for train, test in LeaveOneOut(Ys.shape[0]):
    lm = Lasso().fit(Xs[train], Ys[train])
    score = np.linalg.norm(lm.predict(Xs[test]) - Ys[test])
    scores.append((score, strains[test[0]]))
print '[INFO] Model training done'

scores = DataFrame(scores, columns=['error', 'strain']).sort(columns='error')
scores['strain_name'] = [acc_name.loc[x, 'gene'].split(';')[0] for x in scores['strain']]

scores.to_csv(wd + 'tables/lm_growth_prediction.tab', index=False, sep='\t')
print '[INFO] Growth prediction done!'


# Kinases -> metabolites
X, Y = kinase_df.copy(), metabol_df.copy()

pred_df, meas_df, error_df, m_features = [], [], [], dict()
models = {}
for i in range(Y.shape[0]):
    Xs, Ys = X.T.copy().replace(np.NaN, 0.0), Y.ix[i].copy()

    samples, features, metabolite = Xs.index, Xs.columns, Y.index[i]

    m_features[metabolite] = []
    models[metabolite] = {}

    cv = LeaveOneOut(len(samples))
    for train, test in cv:
        train_i, test_i = samples[train], samples[test]

        fs = SelectKBest(f_regression, 10).fit(Xs.ix[train_i], Ys[train_i])
        m_features[metabolite].extend(features[fs.get_support()])

        lm = LinearRegression(normalize=True).fit(Xs.ix[train_i], Ys[train_i])

        pred = lm.predict(Xs.ix[test_i])[0]

        meas = Ys[test_i][0]
        error = np.linalg.norm(pred - meas)

        pred_df.append((metabolite, test_i[0], pred))
        meas_df.append((metabolite, test_i[0], meas))
        error_df.append((metabolite, test_i[0], error))

        models[metabolite][test_i[0]] = lm

        print metabolite, test_i[0], pred, meas, error

error_df = pivot_table(DataFrame(error_df, columns=['metabolite', 'strain', 'value']), values='value', index='strain', columns='metabolite')
meas_df = pivot_table(DataFrame(meas_df, columns=['metabolite', 'strain', 'value']), values='value', index='strain', columns='metabolite')
pred_df = pivot_table(DataFrame(pred_df, columns=['metabolite', 'strain', 'value']), values='value', index='strain', columns='metabolite')

m_features_df = {k: dict(zip(*np.unique(v, return_counts=True))) for k, v in m_features.items()}
m_features_df = DataFrame(m_features_df).replace(np.NaN, 0.0)

error_df.to_csv(wd + 'tables/lm_error.tab', sep='\t')
meas_df.to_csv(wd + 'tables/lm_measured.tab', sep='\t')
pred_df.to_csv(wd + 'tables/lm_predicted.tab', sep='\t')
m_features_df.to_csv(wd + 'tables/lm_features.tab', sep='\t')
print '[INFO] Model training done'


# Prediction of the dynamic data-set
X, Y = kinase_df.dropna().copy().T, metabol_df.dropna().copy().T
X_test, Y_test = dyn_kinase_df.dropna().copy().T, dyn_metabol_df.dropna().copy().T

# Centre features
X, X_test = zscore(X), zscore(X_test)

# Sort and re-shape data-sets
dyncond, kinases, metabol = Y_test.index.values, list(set(X.columns).intersection(X_test.columns)), list(set(Y.columns).intersection(Y_test.columns))

X, Y = X.ix[strains, kinases], Y.ix[strains, metabol]
X_test, Y_test = X_test.ix[dyncond, kinases], Y_test.ix[dyncond, metabol]
print '[INFO] %d kinases, %d metabolites' % (len(kinases), len(metabol))

# Run linear models
models = {m: LinearRegression().fit(X, Y[m]) for m in metabol}

# Run predictions
Y_test_predict = DataFrame({m: models[m].predict(X_test) for m in metabol}, index=dyncond)

# Predict conditions across metabolites
cor_pred_c = [(c, pearsonr(Y_test.ix[c, metabol], Y_test_predict.ix[c, metabol]), Y_test.ix[c, metabol].values, Y_test_predict.ix[c, metabol].values) for c in dyncond]
cor_pred_c = DataFrame([(m, c, p, x[i], y[i]) for m, (c, p), x, y in cor_pred_c for i in range(len(x))], columns=['cond', 'cor', 'pvalue', 'y_true', 'y_pred'])

titles = {k: '%s (r=%.2f)' % (k, c) for k, c in cor_pred_c.groupby('cond').first()['cor'].to_dict().items()}
g = sns.lmplot('y_true', 'y_pred', cor_pred_c, col='cond', col_wrap=6, size=3, scatter_kws={'s': 50, 'alpha': .8}, line_kws={'c': '#FFFFFF', 'alpha': .7}, ci=90, palette='muted', sharex=False, sharey=False, col_order=dyncond)
[ax.set_title(titles[title]) for ax, title in zip(g.axes.flat, dyn_metabol_df.columns)]
plt.savefig(wd + 'reports/%s_lm_pred_conditions.pdf' % version, bbox_inches='tight')
plt.close('all')

# Predict metabolites across conditions
cor_pred_m = [(m, pearsonr(Y_test.ix[dyncond, m], Y_test_predict.ix[dyncond, m]), Y_test.ix[dyncond, m].values, Y_test_predict.ix[dyncond, m].values) for m in metabol]
cor_pred_m = DataFrame([(m, c, p, x[i], y[i]) for m, (c, p), x, y in cor_pred_m for i in range(len(x))], columns=['met', 'cor', 'pvalue', 'y_true', 'y_pred'])

col_order = cor_pred_m.groupby('met').first().sort('cor', ascending=False).index
titles = {k: '%s (r=%.2f)' % (k, c) for k, c in cor_pred_m.groupby('met').first()['cor'].to_dict().items()}
g = sns.lmplot('y_true', 'y_pred', cor_pred_m, col='met', col_wrap=8, size=3, scatter_kws={'s': 50, 'alpha': .8}, line_kws={'c': '#FFFFFF', 'alpha': .7}, ci=90, palette='muted', sharex=False, sharey=False, col_order=col_order)
[ax.set_title(titles[title]) for ax, title in zip(g.axes.flat, col_order)]
plt.savefig(wd + 'reports/%s_lm_pred_metabolites.pdf' % version, bbox_inches='tight')
plt.close('all')

print '[INFO] Predictions done'