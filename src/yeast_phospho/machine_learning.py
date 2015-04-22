import numpy as np
from pandas import DataFrame, read_csv, Index, pivot_table
from sklearn.cross_validation import KFold, LeaveOneOut
from sklearn.feature_selection.univariate_selection import f_classif, SelectKBest, chi2, f_regression
from sklearn.linear_model import Lasso, LinearRegression


wd = '/Users/emanuel/Projects/projects/yeast_phospho/'

# Import kinase enrichment
kinase_df = read_csv(wd + 'tables/kinase_enrichment.tab', sep='\t', index_col=0)

# Import metabol log2 FC
metabol_df = read_csv(wd + 'tables/steady_state_metabolomics.tab', sep='\t', index_col=0)
metabol_df.index = Index(metabol_df.index, dtype=str)

# Import growth rates
growth = read_csv(wd + 'files/strain_relative_growth_rate.txt', sep='\t', index_col=0)

# Import acc map to name form uniprot
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)

# Overlapping kinases/phosphatases knockout
strains = list(set(kinase_df.columns).intersection(set(metabol_df.columns)))

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

# # Test case
# kinases_targets_test = {k: t.intersection(phospho_df.index) for k, t in kinases_targets.items() if len(t.intersection(phospho_test_df.index)) > 2}
# kinase_test_df = [(k, gsea(phospho_test_df.to_dict()['fc'], targets, True, 1000)[:2]) for k, targets in kinases_targets_test.items()]
# kinase_test_df = Series({k: -np.log10(pvalue) if es < 0 else np.log10(pvalue) for k, (es, pvalue) in kinase_test_df})
#
# for m in set(metabol_test_df.index).intersection(metabol_df.index):
#     Xs, Ys = X.T.copy().replace(np.NaN, 0.0), Y.ix[m].copy()
#
#     lm = LinearRegression(normalize=True).fit(Xs, Ys)
#
#     Xs_test, Ys_test = [kinase_test_df[k] if k in kinase_test_df else 0.0 for k in kinase_df.index], metabol_test_df[m]
#     print lm.predict(Xs_test), Ys_test