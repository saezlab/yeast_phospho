import numpy as np
from sklearn.cross_validation import LeaveOneOut
from statsmodels.stats.multitest import multipletests
from yeast_phospho import wd
from yeast_phospho.utils import pearson
from sklearn.linear_model import LinearRegression
from pandas import DataFrame, read_csv


lm = LinearRegression()

r_activity = read_csv('%s/tables/reaction_activity_steady_state.tab' % wd, sep='\t', index_col=0)
r_activity_growth = read_csv('%s/tables/reaction_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0)


# ---- Steady-state: predict reaction activity
print '[INFO] Steady-state reaction activity predictions with Kinase activities'

k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)
k_activity_growth = read_csv('%s/tables/kinase_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0)

strains, kinases, reactions = list(k_activity.columns), list(k_activity.index), list(r_activity.index)
for xs, ys in [(k_activity, r_activity), (k_activity_growth, r_activity_growth)]:
    x, y = xs.loc[kinases, strains].replace(np.NaN, 0.0).T, ys.loc[reactions, strains].T

    r_predicted = DataFrame({strains[test]: dict(zip(*(reactions, lm.fit(x.ix[train], y.ix[train, reactions]).predict(x.ix[test])[0]))) for train, test in LeaveOneOut(len(x))})

    # Plot predicted prediction scores
    r_score = [(r, pearson(ys.ix[r, strains], r_predicted.ix[r, strains])) for r in reactions]
    r_score = DataFrame([(m, c, p, n) for m, (c, p, n) in r_score], columns=['reaction', 'correlation', 'pvalue', 'n_meas'])
    r_score['adjpvalue'] = multipletests(r_score['pvalue'], method='fdr_bh')[1]
    r_score = r_score.sort('correlation', ascending=False)
    print 'Mean correlation metabolites: ', r_score['correlation'].mean()

    s_score = [(s, pearson(ys.ix[reactions, s], r_predicted.ix[reactions, s])) for s in strains]
    s_score = DataFrame([(m, c, p, n) for m, (c, p, n) in s_score], columns=['strain', 'correlation', 'pvalue', 'n_meas'])
    s_score['adjpvalue'] = multipletests(s_score['pvalue'], method='fdr_bh')[1]
    s_score = s_score.sort('correlation', ascending=False)
    print 'Mean correlation samples: ', s_score['correlation'].mean()


# ---- Steady-state: predict metabolites FC with TF activity
print '[INFO] Steady-state reaction activity predictions with TF activities'

tf_activity = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0)
tf_activity_growth = read_csv('%s/tables/tf_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0)

strains, tfs, reactions = list(set(tf_activity.columns).intersection(k_activity.columns)), list(tf_activity.index), list(r_activity.index)
for xs, ys in [(tf_activity, r_activity), (tf_activity_growth, r_activity_growth)]:
    x, y = xs.loc[:, strains].dropna().T, ys.loc[reactions, strains].T

    r_predicted = DataFrame({strains[test]: dict(zip(*(reactions, lm.fit(x.ix[train], y.ix[train, reactions]).predict(x.ix[test])[0]))) for train, test in LeaveOneOut(len(x))})

    # Plot predicted prediction scores
    r_score = [(r, pearson(ys.ix[r, strains], r_predicted.ix[r, strains])) for r in reactions]
    r_score = DataFrame([(m, c, p, n) for m, (c, p, n) in r_score], columns=['reaction', 'correlation', 'pvalue', 'n_meas'])
    r_score = r_score.set_index('reaction')
    r_score['adjpvalue'] = multipletests(r_score['pvalue'], method='fdr_bh')[1]
    r_score['signif'] = ['FDR < 5%' if i < 0.05 else ' FDR >= 5%' for i in r_score['adjpvalue']]
    r_score = r_score.sort('correlation', ascending=False)
    print 'Mean correlation metabolites: ', r_score['correlation'].mean()

    s_score = [(s, pearson(ys.ix[reactions, s], r_predicted.ix[reactions, s])) for s in strains]
    s_score = DataFrame([(m, c, p, n) for m, (c, p, n) in s_score], columns=['strain', 'correlation', 'pvalue', 'n_meas'])
    s_score = s_score.set_index('strain')
    s_score['adjpvalue'] = multipletests(s_score['pvalue'], method='fdr_bh')[1]
    s_score['signif'] = ['FDR < 5%' if i < 0.05 else ' FDR >= 5%' for i in s_score['adjpvalue']]
    s_score = s_score.sort('correlation', ascending=False)
    print 'Mean correlation samples: ', s_score['correlation'].mean()


# ---- Dynamic: predict metabolites FC with kinases
print '[INFO] Dynamic reaction activity predictions with Kinase activities'

k_activity_dyn = read_csv(wd + 'tables/kinase_activity_dynamic.tab', sep='\t', index_col=0)
r_activity_dyn = read_csv('%s/tables/reaction_activity_dynamic.tab' % wd, sep='\t', index_col=0)

for xs, ys in [(k_activity, r_activity), (k_activity_growth, r_activity_growth)]:
    # Compute overlaps
    kinases_dyn = list(set(k_activity_dyn.index).intersection(xs.index))
    conditions_dyn = list(k_activity_dyn.columns)
    reactions_dyn = list(set(r_activity_dyn.index).intersection(ys.index))

    # Fit linear regression model
    x_train, y_train = xs.ix[kinases_dyn, strains].replace(np.NaN, 0.0).T, ys.ix[reactions_dyn, strains].T
    x_test, y_test = k_activity_dyn.ix[kinases_dyn, conditions_dyn].replace(np.NaN, 0.0).T, r_activity_dyn.ix[reactions_dyn, conditions_dyn].T

    y_pred = DataFrame(lm.fit(x_train, y_train).predict(x_test), index=conditions_dyn, columns=reactions_dyn)

    # Plot predicted prediction scores
    r_score_dyn = [(r, pearson(y_test.ix[conditions_dyn, r], y_pred.ix[conditions_dyn, r])) for r in reactions_dyn]
    r_score_dyn = DataFrame([(m, c, p, n) for m, (c, p, n) in r_score_dyn], columns=['reaction', 'correlation', 'pvalue', 'n_meas']).dropna()
    r_score_dyn = r_score_dyn.set_index('reaction')
    r_score_dyn['adjpvalue'] = multipletests(r_score_dyn['pvalue'], method='fdr_bh')[1]
    r_score_dyn['signif'] = ['FDR < 5%' if x < 0.05 else ' FDR >= 5%' for x in r_score_dyn['adjpvalue']]
    r_score_dyn = r_score_dyn.sort('correlation', ascending=False)
    print 'Mean correlation metabolites: ', r_score_dyn['correlation'].mean()

    s_score_dyn = [(s, pearson(y_test.ix[s, reactions_dyn], y_pred.ix[s, reactions_dyn])) for s in conditions_dyn]
    s_score_dyn = DataFrame([(m, c, p, n) for m, (c, p, n) in s_score_dyn], columns=['condition', 'correlation', 'pvalue', 'n_meas']).dropna()
    s_score_dyn = s_score_dyn.set_index('condition')
    s_score_dyn['adjpvalue'] = multipletests(s_score_dyn['pvalue'], method='fdr_bh')[1]
    s_score_dyn['signif'] = ['FDR < 5%' if x < 0.05 else ' FDR >= 5%' for x in s_score_dyn['adjpvalue']]
    s_score_dyn = s_score_dyn.sort('correlation', ascending=False)
    print 'Mean correlation samples: ', s_score_dyn['correlation'].mean()
