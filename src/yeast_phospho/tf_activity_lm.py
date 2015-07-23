import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from yeast_phospho.utils import pearson
from sklearn.cross_validation import KFold
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
from sklearn.metrics.regression import mean_squared_error
from pandas import DataFrame, read_csv, pivot_table

sns.set_style('ticks')

# Import growth rates
growth = read_csv(wd + 'files/strain_relative_growth_rate.txt', sep='\t', index_col=0)['relative_growth']

# Import conversion table
name2id = read_csv(wd + 'tables/orf_name_dataframe.tab', sep='\t', index_col=1).to_dict()['orf']
id2name = read_csv(wd + 'tables/orf_name_dataframe.tab', sep='\t', index_col=0).to_dict()['name']

# Import gene-expression data-set
gexp = read_csv(wd + 'data/gene_expresion/tf_ko_gene_expression.tab', sep='\t', header=False)
gexp = gexp[gexp['study'] == 'Kemmeren_2014']
gexp['tf'] = [name2id[i] if i in name2id else id2name[i] for i in gexp['tf']]
gexp = pivot_table(gexp, values='value', index='target', columns='tf')
print '[INFO] Gene-expression imported!'

# Overlap conditions with growth measurements
strains = list(set(gexp.columns).intersection(growth.index))

# Filter gene-expression to overlapping conditions
gexp = gexp[strains]
gexp = gexp[(gexp.abs() > 2).sum(1) > 1]

# Export GEX
gexp.to_csv(wd + 'data/steady_state_transcriptomics.tab', sep='\t')

# TF targets
tf_targets = read_csv(wd + 'data/tf_network/tf_gene_network_chip_only.tab', sep='\t')
tf_targets['tf'] = [name2id[i] if i in name2id else id2name[i] for i in tf_targets['tf']]
tf_targets['interaction'] = 1
tf_targets = pivot_table(tf_targets, values='interaction', index='target', columns='tf', fill_value=0)
print '[INFO] TF targets calculated!'


def calculate_activity(strain):
    y = gexp.ix[tf_targets.index, strain].dropna()
    x = tf_targets.ix[y.index]

    x = x.loc[:, x.sum() != 0]

    best_model = (np.Inf, 0.0)
    for train, test in KFold(len(x), 5):
        lm = RidgeCV().fit(x.ix[train], y.ix[train])
        score = mean_squared_error(lm.predict(x.ix[test]), y.ix[test].values)

        if score < best_model[0]:
            best_model = (score, lm.alpha_, lm.coef_)

    print '[INFO] %s, score: %.3f, alpha: %.2f' % (strain, best_model[0], best_model[1])

    return dict(zip(*(x.columns, Ridge(alpha=best_model[0]).fit(x, y).coef_)))

tf_activity = DataFrame({c: calculate_activity(c) for c in strains})
print '[INFO] Kinase activity calculated: ', tf_activity.shape

# Regress out growth


def regress_out_growth(kinase):
    x, y = growth.ix[strains].values, tf_activity.ix[kinase, strains].values

    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))

    if sum(mask) > 3:
        x, y = x[mask], y[mask]

        lm = LinearRegression().fit(np.mat(x).T, y)

        y_ = y - lm.coef_[0] * x - lm.intercept_

        return dict(zip(np.array(strains)[mask], y_))

    else:
        return {}

tf_activity = DataFrame({kinase: regress_out_growth(kinase) for kinase in tf_activity.index}).T.dropna(axis=0, how='all')
print '[INFO] Growth regressed out from the Kinases activity scores: ', tf_activity.shape

# Export kinase activity matrix
tf_activity_file = '%s/tables/tf_activity_steady_state.tab' % wd
tf_activity.to_csv(tf_activity_file, sep='\t')
print '[INFO] [TF ACTIVITY] Exported to: %s' % tf_activity_file
