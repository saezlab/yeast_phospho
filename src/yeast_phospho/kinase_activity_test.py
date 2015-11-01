from __future__ import division
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas.stats.misc import zscore
from sklearn.linear_model import Ridge
from pandas import DataFrame, Series, read_csv, melt, pivot_table, concat
from yeast_phospho.utilities import pearson, get_kinases_targets, get_protein_sequence
from yeast_phospho.utilities import read_fasta, flanking_sequence, position_weight_matrix, similarity_score_matrix, AA_PRIORS_YEAST

acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=0)
acc_name.index = [i.split(';')[0] for i in acc_name.index]
acc_name = acc_name['oln'].to_dict()

# ---- Import data
# data = read_csv('%s/tables/pproteomics_dynamic.tab' % wd, sep='\t', index_col=0)
data = read_csv('%s/tables/pproteomics_steady_state.tab' % wd, sep='\t', index_col=0)
data_proteins = {i.split('_')[0] for i in data.index}

sequences = read_fasta('%s/files/orf_trans_all.fasta' % wd)

stringdb = read_csv('/Users/emanuel/Downloads/4932.protein.links.v10.txt', sep=' ')
stringdb = stringdb[stringdb['combined_score'] > 500]
stringdb['protein1'] = [i.split('.')[1] for i in stringdb['protein1']]
stringdb['protein2'] = [i.split('.')[1] for i in stringdb['protein2']]
stringdb = stringdb[[p1 in data_proteins and p2 in data_proteins for p1, p2 in stringdb[['protein1', 'protein2']].values]]
stringdb = stringdb[[p1 in sequences and p2 in sequences for p1, p2 in stringdb[['protein1', 'protein2']].values]]


phosphogrid = get_kinases_targets()

flanking_targets_all = flanking_sequence(sequences, {i for i in data.index if len(i.split('_')) == 2}, flank=7)


def calculate_ssm(kinase):
    psites = set(phosphogrid[phosphogrid[kinase] != 0].index)

    flanking_targets = flanking_sequence(sequences, psites, flank=7)

    pwm, ic = position_weight_matrix(flanking_targets, AA_PRIORS_YEAST)

    interactions = {list(set([p1, p2]) - set([kinase]))[0] for p1, p2 in stringdb[['protein1', 'protein2']].values if kinase in [p1, p2]}

    ssm = similarity_score_matrix({p: flanking_targets_all[p] for p in flanking_targets_all if p.split('_')[0] in interactions}, pwm, ic)

    if len(ssm) != 0:
        ssm.index = ssm.index.droplevel(1)
        return ssm.to_dict()

    else:
        return {}

k_targets = DataFrame({k: calculate_ssm(k) for k in phosphogrid}).replace(np.NaN, .0)
print '[INFO] Similarity matrix calculated!'


# # ---- Calculate kinase activity with Sklearn
# def k_activity_with_sklearn(x, y):
#     ys = y.ix[x.index].dropna()
#     xs = x.ix[ys.index]
#
#     xs = xs.loc[:, xs.sum() != 0]
#
#     lm = Ridge().fit(xs, zscore(ys))
#
#     return dict(zip(*(xs.columns, lm.coef_)))
#
# k_activity_sklearn = DataFrame({c: k_activity_with_sklearn(k_targets, data[c]) for c in data})


# ---- Calculate kinase activity with Statsmodel
def k_activity_with_statsmodel(x, y):
    ys = y.dropna()
    xs = x.ix[ys.index].replace(np.NaN, 0.0)

    xs = xs.loc[:, xs.sum() != 0]
    xs['const'] = 1

    lm = sm.OLS(zscore(ys), xs)

    res = lm.fit_regularized(L1_wt=0)

    print res.summary()

    return res.params.drop('const').to_dict()

k_activity_statsmodel = DataFrame({c: k_activity_with_statsmodel(k_targets, data[c]) for c in data})

# print '[INFO] Estimations done'
#
# k_activity_statsmodel.ix['YJR066W'].to_csv('/Users/emanuel/Downloads/binary_without_2_papers.csv')
#
# x = read_csv('/Users/emanuel/Downloads/binary_without_2_papers.csv', index_col=0, header=None, names=['activities'])
# y = read_csv('/Users/emanuel/Downloads/similarity_with_2_papers.csv', index_col=0, header=None, names=['activities'])
# z = read_csv('/Users/emanuel/Downloads/similarity_without_2_papers.csv', index_col=0, header=None, names=['activities'])
#
# dyn_xorder = [
#     'N_downshift_5min', 'N_downshift_9min', 'N_downshift_15min', 'N_downshift_25min', 'N_downshift_44min', 'N_downshift_79min',
#     'N_upshift_5min', 'N_upshift_9min', 'N_upshift_15min', 'N_upshift_25min', 'N_upshift_44min', 'N_upshift_79min',
#     'Rapamycin_5min', 'Rapamycin_9min', 'Rapamycin_15min', 'Rapamycin_25min', 'Rapamycin_44min', 'Rapamycin_79min'
# ]
#
# df = concat([x.ix[dyn_xorder, 'activities'], y.ix[dyn_xorder, 'activities'], z.ix[dyn_xorder, 'activities']], axis=1, keys=['binary', 'similarity', 'similarity_full'])
#
# sns.set(style='ticks')
# plt.plot(df, ls='--', marker='o')

# ---- Plot
# ---- Import YeastGenome gene annotation
k_activity = k_activity_statsmodel.copy()

gene_annotation = read_csv('%s/files/gene_association.txt' % wd, sep='\t', header=None).dropna(how='all', axis=1)
gene_annotation['gene'] = [i.split('|')[0] if str(i) != 'nan' else '' for i in gene_annotation[10]]
gene_annotation = gene_annotation.groupby('gene').first()

kinases_type = DataFrame([(i, gene_annotation.ix[i, 9]) for i in k_activity.index], columns=['name', 'info'])
kinases_type['type'] = ['Kinase' if 'kinase' in i.lower() else 'Phosphatase' if 'phosphatase' in i.lower() else 'ND' for i in kinases_type['info']]
kinases_type = kinases_type.set_index('name')


plot_df = k_activity.copy()
plot_df.columns.name = 'strain'
plot_df['kinase'] = plot_df.index
plot_df = melt(plot_df, id_vars='kinase', value_name='activity')
plot_df['type'] = [kinases_type.ix[i, 'type'] for i in plot_df['kinase']]
plot_df['diagonal'] = ['KO' if k == s else 'WT' for k, s in plot_df[['kinase', 'strain']].values]
plot_df['#targets'] = [len(set(k_targets[k_targets[k] == 1].index).intersection(data.index)) for k in plot_df['kinase']]
plot_df['#targets'] = [str(k) if k <= 10 else '> 10' for k in plot_df['#targets']]


hue_order = ['KO', 'WT']

sns.set(style='ticks', palette='pastel')
g = sns.FacetGrid(data=plot_df, legend_out=True, sharey=False, size=4, aspect=.7)
g.map(plt.axhline, y=0, ls=':', c='.5')
g.map(sns.boxplot, 'type', 'activity', 'diagonal', palette={'KO': '#e74c3c', 'WT': '#95a5a6'}, hue_order=hue_order, sym='', width=.5)
g.map(sns.stripplot, 'type', 'activity', 'diagonal', palette={'KO': '#e74c3c', 'WT': '#95a5a6'}, hue_order=hue_order, jitter=True, size=5)
g.add_legend()
g.set_axis_labels('', 'betas')
sns.despine(trim=True)
g.set_xticklabels(rotation=50)
plt.savefig('%s/reports/kinase_activity_test.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'

# # ---- Clustermap
# cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)
# sns.set(style='white', palette='pastel', context='paper')
# sns.clustermap(k_activity.replace(np.NaN, 0.0), robust=True, cmap=cmap, figsize=(5, 20))
# plt.savefig('%s/reports/kinase_activity_test_clustermap.pdf' % wd, bbox_inches='tight')
# plt.close('all')
# print '[INFO] Plot done'
