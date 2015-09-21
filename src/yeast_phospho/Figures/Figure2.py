import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from yeast_phospho import wd
from yeast_phospho.utils import pearson
from matplotlib.gridspec import GridSpec
from sklearn.cross_validation import LeaveOneOut
from sklearn.linear_model import Lasso
from pandas import DataFrame, read_csv


m_signif = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)
m_signif = list(m_signif[m_signif.std(1) > .4].index)

k_signif = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)
k_signif = list(k_signif[(k_signif.count(1) / k_signif.shape[1]) > .75].index)

tf_signif = list(read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0).dropna().index)


# ---- Import
# Steady-state
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0).ix[m_signif]
k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0).ix[k_signif].replace(np.NaN, 0.0)
tf_activity = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0).ix[tf_signif].dropna()

metabolomics_g = read_csv('%s/tables/metabolomics_steady_state_growth_rate.tab' % wd, sep='\t', index_col=0).ix[m_signif]
k_activity_g = read_csv('%s/tables/kinase_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0).ix[k_signif].replace(np.NaN, 0.0)
tf_activity_g = read_csv('%s/tables/tf_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0).ix[tf_signif].dropna()

# Dynamic
metabolomics_dyn = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0).ix[m_signif].dropna()
k_activity_dyn = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0).ix[k_signif].dropna(how='all').replace(np.NaN, 0.0)
tf_activity_dyn = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0).ix[tf_signif].dropna()


# ---- Overlap
strains = list(set(metabolomics.columns).intersection(k_activity.columns).intersection(tf_activity.columns))
conditions = list(set(metabolomics_dyn.columns).intersection(k_activity_dyn.columns).intersection(tf_activity_dyn.columns))
metabolites = list(set(metabolomics.index).intersection(metabolomics_dyn.index))
kinases = list(set(k_activity.index).intersection(k_activity_dyn.index))
tfs = list(set(tf_activity.index).intersection(tf_activity_dyn.index))

metabolomics, k_activity, tf_activity = metabolomics.ix[metabolites, strains], k_activity.ix[kinases, strains], tf_activity.ix[tfs, strains]
metabolomics_g, k_activity_g, tf_activity_g = metabolomics_g.ix[metabolites, strains], k_activity_g.ix[kinases, strains], tf_activity_g.ix[tfs, strains]
metabolomics_dyn, k_activity_dyn, tf_activity_dyn = metabolomics_dyn.ix[metabolites, conditions], k_activity_dyn.ix[kinases, conditions], tf_activity_dyn.ix[tfs, conditions]

k_tf_activity = k_activity.append(tf_activity)
k_tf_activity_g = k_activity_g.append(tf_activity_g)
k_tf_activity_dyn = k_activity_dyn.append(tf_activity_dyn)


# ---- Perform predictions
# Steady-state comparisons
steady_state = [
    (k_activity.copy(), metabolomics.copy(), 'LOO', 'kinase', 'no growth'),
    (tf_activity.copy(), metabolomics.copy(), 'LOO', 'tf', 'no growth'),
    (k_tf_activity.copy(), metabolomics.copy(), 'LOO', 'overlap', 'no growth'),

    (k_activity_g.copy(), metabolomics_g.copy(), 'LOO', 'kinase', 'with growth'),
    (tf_activity_g.copy(), metabolomics_g.copy(), 'LOO', 'tf', 'with growth'),
    (k_tf_activity_g.copy(), metabolomics_g.copy(), 'LOO', 'overlap', 'with growth'),

    (k_activity_dyn.copy(), metabolomics_dyn.copy(), 'LOO', 'kinase', 'dynamic'),
    (tf_activity_dyn.copy(), metabolomics_dyn.copy(), 'LOO', 'tf', 'dynamic'),
    (k_tf_activity_dyn.copy(), metabolomics_dyn.copy(), 'LOO', 'overlap', 'dynamic')
]

# Dynamic comparisons
test_comparisons = [
    ((k_activity, metabolomics), (k_activity_dyn, metabolomics_dyn), 'Test', 'kinase', 'no growth'),
    ((tf_activity, metabolomics), (tf_activity_dyn, metabolomics_dyn), 'Test', 'tf', 'no growth'),
    ((k_tf_activity, metabolomics), (k_tf_activity_dyn, metabolomics_dyn), 'Test', 'overlap', 'no growth'),

    ((k_activity_g, metabolomics_g), (k_activity_dyn, metabolomics_dyn), 'Test', 'kinase', 'with growth'),
    ((tf_activity_g, metabolomics_g), (tf_activity_dyn, metabolomics_dyn), 'Test', 'tf', 'with growth'),
    ((k_tf_activity_g, metabolomics_g), (k_tf_activity_dyn, metabolomics_dyn), 'Test', 'overlap', 'with growth')
]

# Dynamic comparisons
dynamic = [
    ((k_activity.copy(), metabolomics.copy()), (k_activity_dyn.copy(), metabolomics_dyn.copy()), 'Dynamic', 'kinase', 'no growth'),
    ((tf_activity.copy(), metabolomics.copy()), (tf_activity_dyn.copy(), metabolomics_dyn.copy()), 'Dynamic', 'tf', 'no growth'),
    ((k_tf_activity.copy(), metabolomics.copy()), (k_tf_activity_dyn.copy(), metabolomics_dyn.copy()), 'Dynamic', 'overlap', 'no growth'),

    ((k_activity_g.copy(), metabolomics_g.copy()), (k_activity_dyn.copy(), metabolomics_dyn.copy()), 'Dynamic', 'kinase', 'with growth'),
    ((tf_activity_g.copy(), metabolomics_g.copy()), (tf_activity_dyn.copy(), metabolomics_dyn.copy()), 'Dynamic', 'tf', 'with growth'),
    ((k_tf_activity_g.copy(), metabolomics_g.copy()), (k_tf_activity_dyn.copy(), metabolomics_dyn.copy()), 'Dynamic', 'overlap', 'with growth')
]


lm, lm_res, pred = Lasso(alpha=0.01, max_iter=2000), [], {}

for xs, ys, condition, feature, growth in steady_state:
    x_features, y_features, samples = list(xs.index), list(ys.index), list(set(xs.columns).intersection(ys.columns))

    x, y = xs.ix[x_features, samples].replace(np.NaN, 0.0).T, ys.ix[y_features, samples].T

    cv = LeaveOneOut(len(samples))
    y_pred = DataFrame({samples[test]: {y_feature: lm.fit(x.ix[train], y.ix[train, y_feature]).predict(x.ix[test])[0] for y_feature in y_features} for train, test in cv})
    pred['_'.join([condition, feature, growth])] = y_pred.copy()

    lm_res.extend([(condition, feature, growth, f, 'features', pearson(ys.ix[f, samples], y_pred.ix[f, samples])[0]) for f in y_features])
    lm_res.extend([(condition, feature, growth, s, 'samples', pearson(ys.ix[y_features, s], y_pred.ix[y_features, s])[0]) for s in samples])

    print '[INFO] %s, %s, %s' % (condition, feature, growth)
    print '[INFO] x_features: %d, y_features: %d, samples: %d' % (len(x_features), len(y_features), len(samples))


for (xs_train, ys_train), (xs_test, ys_test), condition, feature, growth in dynamic:
    x_features, y_features = list(set(xs_train.index).intersection(xs_test.index)), list(set(ys_train.index).intersection(ys_test.index))
    train_samples, test_samples = list(set(xs_train.columns).intersection(ys_train.columns)), list(set(xs_test.columns).intersection(ys_test.columns))

    x_train, y_train = xs_train.ix[x_features, train_samples].replace(np.NaN, 0.0).T, ys_train.ix[y_features, train_samples].T
    x_test, y_test = xs_test.ix[x_features, test_samples].replace(np.NaN, 0.0).T, ys_test.ix[y_features, test_samples].T

    y_pred = DataFrame({y_feature: dict(zip(*(test_samples, lm.fit(x_train, y_train[y_feature]).predict(x_test)))) for y_feature in y_features}).T

    lm_res.extend([(condition, feature, growth, f, 'features', pearson(ys_test.ix[f, test_samples], y_pred.ix[f, test_samples])[0]) for f in y_features])
    lm_res.extend([(condition, feature, growth, s, 'samples', pearson(ys_test.ix[y_features, s], y_pred.ix[y_features, s])[0]) for s in test_samples])

    print '[INFO] %s, %s, %s' % (condition, feature, growth)
    print '[INFO] x_features: %d, y_features: %d, train_samples: %d, test_samples: %d' % (len(x_features), len(y_features), len(train_samples), len(test_samples))


for (xs_train, ys_train), (xs_test, ys_test), condition, feature, growth in test_comparisons:
    x_features, y_features = list(set(xs_train.index).intersection(xs_test.index)), list(set(ys_train.index).intersection(ys_test.index))
    train_samples, test_samples = list(set(xs_train.columns).intersection(ys_train.columns)), list(set(xs_test.columns).intersection(ys_test.columns))

    x_train, y_train = xs_train.ix[x_features, train_samples].replace(np.NaN, 0.0).T, ys_train.ix[y_features, train_samples].T
    x_test = xs_test.ix[x_features, test_samples].replace(np.NaN, 0.0).T

    y_pred = {}
    for y_feature in y_features:
        outlier = y_train[y_feature].abs().argmax()
        y_pred[y_feature] = dict(zip(*(test_samples, lm.fit(x_train.drop(outlier), y_train[y_feature].drop(outlier)).predict(x_test))))

    y_pred = DataFrame(y_pred).T

    lm_res.extend([(condition, feature, growth, f, 'features', pearson(ys_test.ix[f, test_samples], y_pred.ix[f, test_samples])[0]) for f in y_features])
    lm_res.extend([(condition, feature, growth, s, 'samples', pearson(ys_test.ix[y_features, s], y_pred.ix[y_features, s])[0]) for s in test_samples])

    print '[INFO] %s, %s, %s' % (condition, feature, growth)
    print '[INFO] x_features: %d, y_features: %d, train_samples: %d, test_samples: %d' % (len(x_features), len(y_features), len(train_samples), len(test_samples))


lm_res = DataFrame(lm_res, columns=['condition', 'feature', 'growth', 'name', 'type_cor', 'cor'])


# ---- Plot
m_map = read_csv('%s/files/metabolite_mz_map_kegg.txt' % wd, sep='\t')
m_map['mz'] = [float('%.2f' % i) for i in m_map['mz']]
m_map = m_map.drop_duplicates('mz').drop_duplicates('formula')
m_map = m_map.groupby('mz')['name'].apply(lambda i: '; '.join(i)).to_dict()

# Figure 2
dyn_xorder = [
    'N_downshift_5min', 'N_downshift_9min', 'N_downshift_15min', 'N_downshift_25min', 'N_downshift_44min', 'N_downshift_79min',
    'N_upshift_5min', 'N_upshift_9min', 'N_upshift_15min', 'N_upshift_25min', 'N_upshift_44min', 'N_upshift_79min',
    'Rapamycin_5min', 'Rapamycin_9min', 'Rapamycin_15min', 'Rapamycin_25min', 'Rapamycin_44min', 'Rapamycin_79min'
]
palette = {'with growth': '#95a5a6', 'no growth': '#2ecc71', 'dynamic': '#e74c3c'}

condition, growth = 'LOO', 'dynamic'
plot_df = lm_res[lm_res['condition'] == condition]

sns.set(style='ticks')
fig, gs, pos = plt.figure(figsize=(12, 12)), GridSpec(4, 5, width_ratios=[1, 1, .01, 1, 1]), 0
for feature in ['kinase', 'tf']:
    # ---- Features
    sub_plot_df = plot_df[(plot_df['type_cor'] == 'features') & (plot_df['feature'] == feature)]

    # Boxplot
    ax = plt.subplot(gs[pos])
    sns.boxplot('type_cor', 'cor', 'growth', sub_plot_df, palette=palette, sym='', ax=ax)
    sns.stripplot('type_cor', 'cor', 'growth', sub_plot_df, palette=palette, jitter=True, size=5, ax=ax)
    ax.axhline(y=0, ls=':', c='.5')
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_ticks([])
    ax.set_xlabel('')
    ax.set_ylabel('Pearson correlation' if pos == 0 else '')
    ax.set_title('Kinase' if feature == 'kinase' else 'Transcription Factor')
    ax.set_ylim(-1, 1)
    ax.legend().remove()
    sns.despine(trim=True, ax=ax, bottom=True)

    # Scatter
    sub_plot_df = sub_plot_df[sub_plot_df['growth'] == 'dynamic']
    s_feature = sub_plot_df.ix[(sub_plot_df['cor'] - sub_plot_df['cor'].mean()).abs().argmin()]

    ax = plt.subplot(gs[(pos + 5)])
    x, y = metabolomics_dyn.ix[s_feature['name'], dyn_xorder], pred['_'.join([condition, feature, growth])].ix[s_feature['name'], dyn_xorder]
    g = sns.regplot(x, y, scatter_kws={'s': 80}, marker='o', color='#e74c3c', ax=ax)
    g.set_xlabel('Measured')
    ax.set_ylabel('Predicted' if pos == 0 else '')
    ax.set_title('%s\n(%.2f pearson)' % (m_map[s_feature['name']], s_feature['cor']))
    sns.despine(trim=True, ax=ax)

    # ---- Samples
    sub_plot_df = plot_df[(plot_df['type_cor'] == 'samples') & (plot_df['feature'] == feature)]

    # Boxplot
    ax = plt.subplot(gs[pos + 3])
    sns.boxplot('type_cor', 'cor', 'growth', sub_plot_df, palette=palette, sym='', ax=ax)
    sns.stripplot('type_cor', 'cor', 'growth', sub_plot_df, palette=palette, jitter=True, size=5, ax=ax)
    ax.axhline(y=0, ls=':', c='.5')
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_ticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Kinase' if feature == 'kinase' else 'Transcription Factor')
    ax.set_ylim(-1, 1)
    ax.legend().remove()
    sns.despine(trim=True, ax=ax, bottom=True)

    # Scatter
    sub_plot_df = sub_plot_df[sub_plot_df['growth'] == 'dynamic']
    s_feature = sub_plot_df.ix[(sub_plot_df['cor'] - sub_plot_df['cor'].mean()).abs().argmin()]

    ax = plt.subplot(gs[(pos + 8)])
    x, y = metabolomics_dyn[s_feature['name']], pred['_'.join([condition, feature, growth])][s_feature['name']]
    y = y[x.index]
    g = sns.regplot(x, y, scatter_kws={'s': 80}, marker='o', color='#e74c3c', ax=ax)
    g.set_xlabel('Measured')
    g.set_ylabel('')
    ax.set_title('%s\n(%.2f pearson)' % (s_feature['name'], s_feature['cor']))
    sns.despine(trim=True, ax=ax)

    pos += 1

handles = [mlines.Line2D([], [], color=palette[s], label=s) for s in ['no growth', 'with growth', 'dynamic']]
fig.legend(handles, ['no growth', 'with growth', 'dynamic'], loc='upper center', bbox_to_anchor=(0.5, 0.24), fancybox=True, shadow=True, ncol=3)
fig.tight_layout()
plt.savefig('%s/reports/Figure_2.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Figure 2 generated'


# Supplementary material Figure 2
sns.set(style='ticks', palette='pastel', color_codes=True, context='paper')
g = sns.FacetGrid(data=lm_res, col='condition', row='feature', legend_out=True, sharey=True, ylim=(-1, 1), col_order=['LOO', 'Dynamic', 'Test'], size=2.4, aspect=.9)
g.fig.subplots_adjust(wspace=.05, hspace=.05)
g.map(plt.axhline, y=0, ls=':', c='.5')
g.map(sns.boxplot, 'type_cor', 'cor', 'growth', palette={'with growth': '#95a5a6', 'no growth': '#2ecc71', 'dynamic': '#e74c3c'}, sym='')
g.map(sns.stripplot, 'type_cor', 'cor', 'growth', palette={'with growth': '#95a5a6', 'no growth': '#2ecc71', 'dynamic': '#e74c3c'}, jitter=True, size=5)
g.add_legend(title='Growth rate:')
g.set_axis_labels('', 'Correlation (pearson)')
sns.despine(trim=True)
plt.savefig('%s/reports/Figure_Supp_2.pdf' % wd, bbox_inches='tight')
plt.close('all')
