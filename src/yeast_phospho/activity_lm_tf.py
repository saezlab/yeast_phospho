import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from pandas.stats.misc import zscore
from yeast_phospho import wd
from sklearn.linear_model import Ridge, LinearRegression
from pandas import DataFrame, read_csv, pivot_table
from yeast_phospho.utils import pearson

ridge = Ridge(alpha=.1)

# Import growth rates
growth = read_csv('%s/files/strain_relative_growth_rate.txt' % wd, sep='\t', index_col=0)['relative_growth']

# Import conversion table
name2id = read_csv('%s/files/orf_name_dataframe.tab' % wd, sep='\t', index_col=1).to_dict()['orf']
id2name = read_csv('%s/files/orf_name_dataframe.tab' % wd, sep='\t', index_col=0).to_dict()['name']

# TF targets
tf_targets = read_csv('%s/files/tf_gene_network_chip_only.tab' % wd, sep='\t')
tf_targets['tf'] = [name2id[i] if i in name2id else id2name[i] for i in tf_targets['tf']]
tf_targets = tf_targets[tf_targets['tf'] != tf_targets['target']]
tf_targets['interaction'] = 1
tf_targets = pivot_table(tf_targets, values='interaction', index='target', columns='tf', fill_value=0)
print '[INFO] TF targets calculated!'

# ---- Steady-state: gene-expression data-set
gexp = read_csv('%s/data/Kemmeren_2014_zscores_parsed_filtered.tab' % wd, sep='\t', header=False)
gexp['tf'] = [name2id[i] if i in name2id else id2name[i] for i in gexp['tf']]
gexp = pivot_table(gexp, values='value', index='target', columns='tf')
print '[INFO] Gene-expression imported!'

# Overlap conditions with growth measurements
strains = list(set(gexp.columns).intersection(growth.index))

# Filter gene-expression to overlapping conditions
gexp = gexp[strains]

# Export GEX
gexp.to_csv('%s/data/steady_state_transcriptomics.tab' % wd, sep='\t')
print '[INFO] Steady-state transcriptomics exported'


def calculate_activity(strain):
    y = gexp[strain].dropna()
    x = tf_targets.ix[y.index].replace(np.NaN, 0.0)

    x = x.loc[:, x.sum() != 0]

    return dict(zip(*(x.columns, ridge.fit(x, zscore(y)).coef_)))

tf_activity = DataFrame({c: calculate_activity(c) for c in strains})
print '[INFO] TF activity calculated: ', tf_activity.shape

tf_activity.to_csv('%s/tables/tf_activity_steady_state_with_growth.tab' % wd, sep='\t')
print '[INFO] [TF ACTIVITY] Exported '

# Regress out growth


def regress_out_growth(kinase):
    x, y = growth.ix[strains].values, tf_activity.ix[kinase, strains].values

    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))

    x, y = x[mask], y[mask]

    lm = LinearRegression().fit(np.mat(x).T, y)

    y_ = y - lm.coef_[0] * x - lm.intercept_

    return dict(zip(np.array(strains)[mask], y_))

tf_activity = DataFrame({kinase: regress_out_growth(kinase) for kinase in tf_activity.index}).T.dropna(axis=0, how='all')
print '[INFO] Growth regressed out from the Kinases activity scores: ', tf_activity.shape

# Export kinase activity matrix
tf_activity.to_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t')
print '[INFO] [TF ACTIVITY] Exported'

samples = set(tf_activity).intersection(gexp)
plot_df = DataFrame([(i, s, tf_activity.ix[i, s], gexp.ix[i, s]) for i in tf_activity.index if i in gexp.index for s in samples], columns=['TF', 'sample', 'activity', 'expression'])
plot_df = plot_df[[abs(e) >= 2.5 for e in plot_df['expression']]]
plot_df['type'] = ['over' if e > 0 else 'under' for e in plot_df['expression']]

palette = {'under': '#e74c3c', 'over': '#2ecc71'}
sns.set(style='ticks', palette='pastel', color_codes=True, context='paper')
g = sns.FacetGrid(data=plot_df, size=3, aspect=.6)
g.map(plt.axhline, y=0, ls=':', c='.5')
g.map(sns.boxplot, 'type', 'activity', 'type', palette=palette, hue_order=palette.keys(), sym='')
g.map(sns.stripplot, 'type', 'activity', 'type', palette=palette, hue_order=palette.keys(), jitter=True, size=5)
g.set_axis_labels('', 'betas')
sns.despine(trim=True)

handles = [mpatches.Circle((.5, .5), .25, facecolor=v, edgecolor='white', label=k) for k, v in palette.items()]
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=handles)

plt.savefig('%s/reports/tf_activity_expression_validation.pdf' % wd, bbox_inches='tight')
plt.close('all')


samples = set(tf_activity).intersection(gexp)
plot_df = DataFrame([(i, s, tf_activity.ix[i, s], gexp.ix[i, s]) for i in tf_activity.index if i in gexp.index and sum(gexp.ix[i].abs() > 3) > 2 for s in samples], columns=['TF', 'sample', 'activity', 'expression'])

sns.set(style='ticks', color_codes=True, context='paper')
sns.lmplot(data=plot_df, x='expression', y='activity', scatter_kws={'s': 50}, sharey=False, sharex=False, palette='Set1', size=2, aspect=1)
sns.despine(trim=True)
plt.savefig('%s/reports/tf_activity_expression_validation_scatter.pdf' % wd, bbox_inches='tight')
plt.close('all')




# ---- Dynamic: gene-expression data-set
dyn_trans_df = read_csv('%s/tables/transcriptomics_dynamic.tab' % wd, sep='\t', index_col=0)

conditions = list(dyn_trans_df.columns)


def calculate_activity(condition):
    y = dyn_trans_df[condition].dropna()
    x = tf_targets.ix[y.index].replace(np.NaN, 0)

    x = x.loc[:, x.sum() != 0]

    return dict(zip(*(x.columns, ridge.fit(x, zscore(y)).coef_)))

tf_activity = DataFrame({c: calculate_activity(c) for c in conditions})
print '[INFO] TF activity calculated: ', tf_activity.shape

tf_activity.to_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t')
print '[INFO] [TF ACTIVITY] Exported '
