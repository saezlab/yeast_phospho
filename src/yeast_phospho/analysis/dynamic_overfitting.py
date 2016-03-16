import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import ElasticNet
from yeast_phospho.utilities import pearson
from pandas import DataFrame, read_csv, Series, pivot_table
from yeast_phospho.utilities import get_proteins_name, get_metabolites_name


# -- Import IDs maps
acc_name = get_proteins_name()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}

met_name = get_metabolites_name()
met_name = {k: met_name[k] for k in met_name if len(met_name[k].split('; ')) == 1}


# -- Imports
# Metabolomics
metabolomics_dyn_ng = read_csv('%s/tables/metabolomics_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
metabolomics_dyn_ng = metabolomics_dyn_ng[metabolomics_dyn_ng.std(1) > .4]
metabolomics_dyn_ng.index = ['%.4f' % i for i in metabolomics_dyn_ng.index]

# GSEA
k_activity_dyn_ng_gsea = read_csv('%s/tables/kinase_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_ng_gsea = k_activity_dyn_ng_gsea[(k_activity_dyn_ng_gsea.count(1) / k_activity_dyn_ng_gsea.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_dyn_gsea = read_csv('%s/tables/tf_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
tf_activity_dyn_gsea = tf_activity_dyn_gsea[tf_activity_dyn_gsea.std(1) > .4]

# LM
k_activity_dyn_ng_lm = read_csv('%s/tables/kinase_activity_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_ng_lm = k_activity_dyn_ng_lm[(k_activity_dyn_ng_lm.count(1) / k_activity_dyn_ng_lm.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_dyn_ng_lm = read_csv('%s/tables/tf_activity_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
tf_activity_dyn_ng_lm = tf_activity_dyn_ng_lm[tf_activity_dyn_ng_lm.std(1) > .4]

# Conditions
conditions = {'_'.join(c.split('_')[:-1]) for c in metabolomics_dyn_ng}
samples = set(metabolomics_dyn_ng)

comparisons = [
    (k_activity_dyn_ng_gsea, metabolomics_dyn_ng, 'Kinases', 'Gsea'),
    (tf_activity_dyn_gsea, metabolomics_dyn_ng, 'TFs', 'Gsea'),

    (k_activity_dyn_ng_lm, metabolomics_dyn_ng, 'Kinases', 'Lm'),
    (tf_activity_dyn_ng_lm, metabolomics_dyn_ng, 'TFs', 'Lm'),
]

# -- Create lists
tfs_names = {acc_name[i] for i in tf_activity_dyn_gsea.index}
kinases_names = {acc_name[i] for i in k_activity_dyn_ng_gsea.index}

# -- Linear regressions
lm_res, lm_feat = [], []
for (x, y, feature_type, method_type) in comparisons:
    for m in metabolomics_dyn_ng.index:
        for c in conditions:
            ys = y.ix[m, [i for i in y if not i.startswith(c)]]
            xs = x[ys.index].T

            yss = y.ix[m, [i for i in y if i.startswith(c)]]
            xss = x[yss.index].T

            lm = ElasticNet(alpha=0.01).fit(xs, ys)
            pred = Series(lm.predict(xss), index=xss.index)

            features = dict(zip(*(xs.columns, lm.coef_)))
            for f in features:
                lm_feat.append((feature_type, method_type, m, c, f, features[f]))

            lm_res.append((feature_type, method_type, m, c, pearson(yss, pred)[0]))

lm_res = DataFrame(lm_res, columns=['feature', 'method', 'ion', 'condition', 'pearson'])
lm_res['metabolite'] = [met_name[i] for i in lm_res['ion']]
print lm_res.head()

lm_feat = DataFrame(lm_feat, columns=['feature_type', 'method', 'ion', 'condition', 'feature', 'coefficient'])
lm_feat['m_name'] = [met_name[i] for i in lm_feat['ion']]
lm_feat['f_name'] = [acc_name[i] for i in lm_feat['feature']]
print lm_feat.head()


# -- Plot
palette = {'TFs': '#34495e', 'Kinases': '#3498db'}

# Correlation boxplots
for method in ['Gsea', 'Lm']:
    plot_df = lm_res[lm_res['method'] == method]

    sns.set(style='ticks', font_scale=.75, context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
    g = sns.FacetGrid(plot_df, legend_out=True, aspect=1, size=1.5, sharex=True, sharey=True)
    g.map(sns.boxplot, 'pearson', 'condition', 'feature', palette=palette, sym='')
    g.map(sns.stripplot, 'pearson', 'condition', 'feature', palette=palette, jitter=True, size=2, split=True, edgecolor='white', linewidth=.3)
    g.map(plt.axvline, x=0, ls='-', lw=.1)
    plt.xlim([-1, 1])
    g.add_legend()
    g.set_axis_labels('Pearson correlation\n(predicted vs measured)', '')
    g.set_titles(row_template='{row_name}')
    g.fig.subplots_adjust(wspace=.05, hspace=.2)
    sns.despine(trim=True)
    plt.savefig('%s/reports/feature_regression_nitrogen_metabolism_%s.pdf' % (wd, method), bbox_inches='tight')
    plt.close('all')

# Features heatmap
for method in ['Gsea', 'Lm']:
    plot_df = lm_feat[lm_feat['method'] == method]

    plot_df = pivot_table(plot_df, values='coefficient', index='m_name', columns='f_name', aggfunc=np.median)
    plot_df = plot_df.loc[:, plot_df.std() > .1]
    plot_df = plot_df[plot_df.std(1) > .1]

    cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)
    col_c = [palette['TFs'] if c in tfs_names else palette['Kinases'] for c in plot_df]

    sns.set(context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
    g = sns.clustermap(plot_df, figsize=(4, 8), cmap=cmap, linewidth=.5, col_colors=col_c)
    plt.savefig('%s/reports/feature_regression_nitrogen_metabolism_heatmap_%s.pdf' % (wd, method), bbox_inches='tight')
    plt.close('all')

# Barplot + Scatter
sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})
fig, gs = plt.figure(figsize=(5, 8)), GridSpec(4, 2, width_ratios=[1, 1], hspace=0.6, wspace=.5)
for method in ['Gsea', 'Lm']:
    lm_feat_method = lm_feat[lm_feat['method'] == method]
    lm_res_method = lm_res[lm_res['method'] == method]

    coef_table = pivot_table(lm_feat_method, values='coefficient', index='m_name', columns='f_name', aggfunc=np.median)

    # data-sets
    metabolomics = metabolomics_dyn_ng.copy()
    k_activity = k_activity_dyn_ng_gsea.copy() if method == 'Gsea' else k_activity_dyn_ng_lm.copy()
    tf_activity = tf_activity_dyn_gsea.copy() if method == 'Gsea' else tf_activity_dyn_ng_lm.copy()

    metabolomics.index = [met_name[i] for i in metabolomics.index]
    k_activity.index = [acc_name[i] for i in k_activity.index]
    tf_activity.index = [acc_name[i] for i in tf_activity.index]

    # barplot
    order = pivot_table(lm_res_method[lm_res_method['feature'] == 'TFs'], values='pearson', index='metabolite', columns='condition', aggfunc=np.median).median(1)
    order = order[order > .5]
    order = list(order.sort(inplace=False, ascending=False).index)

    plot_df = lm_res_method[[i in order for i in lm_res_method['metabolite']]]

    ax = plt.subplot(gs[:, 0])
    sns.barplot('pearson', 'metabolite', 'feature', plot_df, palette=palette, orient='h', lw=0, ax=ax, order=order, ci=None, estimator=np.median)
    ax.axvline(x=0, ls='-', lw=.1, c='gray')
    ax.set_xlabel('Pearson correlation')
    ax.set_ylabel('Metabolite')
    ax.set_title('Predicted vs Measured')
    ax.set_xlim((0, 1.0))
    sns.despine(ax=ax)
    plt.legend().remove()

    # scatter
    pos = 1
    for m in ['L-Glutamine', 'Guanosine', 'L-Malate', 'L-Arginine']:
        ax = plt.subplot(gs[pos])

        best_tf = coef_table.ix[m, tf_activity.index].abs().argmax()
        best_kinase = coef_table.ix[m, k_activity.index].abs().argmax()

        met_x, kin_y, tfs_y = metabolomics.ix[m, samples], k_activity.ix[best_kinase, samples], tf_activity.ix[best_tf, samples]

        sns.regplot(met_x, kin_y, scatter_kws={'s': 50, 'alpha': .6}, color=palette['Kinases'], label=best_kinase, ax=ax)
        sns.regplot(met_x, tfs_y, scatter_kws={'s': 50, 'alpha': .6}, color=palette['TFs'], label=best_tf, ax=ax)
        sns.despine(ax=ax)
        ax.set_title('%s' % m)
        ax.set_xlabel('Metabolite (log FC)')
        ax.set_ylabel('Kinase/TF activity')
        ax.axhline(0, ls='-', lw=.1, c='gray')
        ax.axvline(0, ls='-', lw=.1, c='gray')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        pos += 2

    plt.savefig('%s/reports/feature_regression_nitrogen_metabolism_barplot+scatter_%s.pdf' % (wd, method), bbox_inches='tight')
    plt.close('all')
    print '[INFO] Figure exported'
