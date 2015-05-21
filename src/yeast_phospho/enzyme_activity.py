import re
import copy
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr
from scipy.stats.distributions import hypergeom
from pymist.reader.sbml_reader import read_sbml_model
from pandas import DataFrame, Series, read_csv, Index, melt, pivot_table
from sklearn.metrics.pairwise import euclidean_distances, linear_kernel


def pearson(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = pearsonr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue


def cohend(x, y):
    return (np.mean(x) - np.mean(y)) / (np.sqrt((np.std(x) ** 2 + np.std(y) ** 2) / 2))


def metric(func, x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    res = func(x[mask], y[mask]) if np.sum(mask) > 5 else np.NaN
    return res if np.isfinite(res) else [[res]]

sns.set_style('white')

# Version
version = 'v1'
print '[INFO] Version: %s' % version

wd = '/Users/emanuel/Projects/projects/yeast_phospho/'

# Import metabol log2 FC
metabol_df = read_csv(wd + 'tables/steady_state_metabolomics.tab', sep='\t', index_col=0)
metabol_df.index = Index(metabol_df.index, dtype=str)

# Import kinase enrichment
kinase_df = read_csv(wd + 'tables/kinase_enrichment_df.tab', sep='\t', index_col=0)

# Overlapping strains
strains = list(set(metabol_df.columns).intersection(kinase_df.columns))
kinase_df = kinase_df[strains]
metabol_df = metabol_df[strains]

# Import metabolites map
m_map = read_csv(wd + 'files/metabolomics/metabolite_mz_map_kegg.txt', sep='\t')  # _adducts
m_map['mz'] = [str(c) for c in m_map['mz']]

# Import metabolic model mapping
model_met_map = read_csv(wd + 'files/metabolomics/metabolite_mz_map_dobson.txt', sep='\t', index_col='id')
model_met_map['mz'] = [str(i) for i in model_met_map['mz']]
model_met_map = model_met_map.drop_duplicates('mz')['mz'].to_dict()

# Import metabolic model
model = read_sbml_model('/Users/emanuel/Projects/resources/metabolic_models/1752-0509-4-145-s1/yeast_4.04.xml')

# Remove extracellular metabolites
s_matrix = model.get_stoichiometric_matrix()
s_matrix = s_matrix[[not i.endswith('_b') for i in s_matrix.index]]

# Remove highly connected metabolites
met_to_remove = [
    'biomass', 'acetyl-CoA', 'carbon dioxide', 'coenzyme A', 'L-glutamate', 'water', 'hydrogen peroxide',
    'H+', 'NAD(+)', 'NADH', 'NADP(+)', 'NADPH', 'ammonium', 'oxygen', 'phosphate', 'diphosphate', '2-oxoglutarate',
    'acyl-CoA', 'ADP', 'AMP', 'ATP', 'UDP', 'UMP', 'UTP', 'CDP', 'CMP', 'CTP', 'GDP', 'GMP', 'GTP',
    'dADP', 'dAMP', 'dATP', 'dUDP', 'dUMP', 'dUTP', 'dCDP', 'dCMP', 'dCTP', 'dGDP', 'dGMP', 'dGTP'
]
met_to_remove = {k for m in met_to_remove for k, v in model.metabolites.items() if re.match('%s \[.*\]' % re.escape(m), v)}
s_matrix = s_matrix[[i not in met_to_remove for i in s_matrix.index]]

# Remove exchange and biomass reactions
reactions_to_remove = np.hstack((model.get_exchanges(True), 'r_1812'))
s_matrix = s_matrix.loc[:, [r not in reactions_to_remove for r in s_matrix.columns]]

# Get reactions products and substrates
r_substrates = {r: set(s_matrix[s_matrix[r] < 0].index) for r in s_matrix.columns}
r_products = {r: set(s_matrix[s_matrix[r] > 0].index) for r in s_matrix.columns}

# Swap stoichiometric values with ones
s_matrix = (s_matrix != 0) + 0

# Remove un-used metabolites and reactions
s_matrix = s_matrix.loc[:, s_matrix.sum() != 0]
s_matrix = s_matrix[s_matrix.sum(1) != 0]

# Calculate metabolites x metabolites distance matrix
_m_distance = s_matrix.dot(s_matrix.T).abs()
_m_distance = (_m_distance != 0) + 0

m_distance = DataFrame(nx.all_pairs_dijkstra_path_length(nx.from_numpy_matrix(_m_distance.values, create_using=nx.DiGraph())))
m_distance.index = _m_distance.index
m_distance.columns = _m_distance.columns
print '[INFO] Metabolites distance calculated!'

# Calculate reactions x reactions distance matrix
_r_distance = s_matrix.T.dot(s_matrix).abs()
_r_distance = (_r_distance != 0) + 0

r_distance = DataFrame(nx.all_pairs_dijkstra_path_length(nx.from_numpy_matrix(_r_distance.values, create_using=nx.DiGraph())))
r_distance.index = _r_distance.index
r_distance.columns = _r_distance.columns
print '[INFO] Metabolic reactions distance calculated!'

# Get reactions metabolites
r_metabolites = {r: set(s_matrix.ix[s_matrix[r] != 0, r].index) for r in s_matrix.columns}

# Reaction neighbour metabolites
distance = 1
r_neighbour_metabolties = {r: set(r_distance.loc[r, r_distance.ix[r] <= distance].index) for r in r_distance.index}
r_neighbour_metabolties = {r: {m for reaction in reactions for m in s_matrix.loc[s_matrix[reaction] != 0, reaction].index} for r, reactions in r_neighbour_metabolties.items()}
r_neighbour_metabolties = {r: {model_met_map[m] for m in metabolites if m in model_met_map} for r, metabolites in r_neighbour_metabolties.items()}
r_neighbour_metabolties = {k: v.intersection(metabol_df.index) for k, v in r_neighbour_metabolties.items()}
r_neighbour_metabolties = {k: v for k, v in r_neighbour_metabolties.items() if len(v) > 1}

# Calculate effect size of reactions
_r_substrates = {r: {model_met_map[s] for s in substrates if s in model_met_map and model_met_map[s] in metabol_df.index} for r, substrates in r_substrates.items()}
_r_substrates = {r: v for r, v in _r_substrates.items() if len(v) > 0}

_r_products = {r: {model_met_map[s] for s in products if s in model_met_map and model_met_map[s] in metabol_df.index} for r, products in r_products.items()}
_r_products = {r: v for r, v in _r_products.items() if len(v) > 0}

r_effect = set(_r_products.keys()).intersection(_r_substrates.keys())
r_effect = [(c, r, metabol_df.loc[_r_products[r], c].mean() - metabol_df.loc[_r_substrates[r], c].mean()) for c in metabol_df.columns for r in r_effect]
r_effect = DataFrame(r_effect, columns=['strain', 'reaction', 'value'])
r_effect = pivot_table(r_effect, values='value', index='reaction', columns='strain')
print '[INFO] Metabolic reactions effect size calculated!'

# Reaction effect size association with metabolic network distance
r_effect_corr = r_effect.T.corr()
r_effect_corr['index'] = r_effect_corr.index
r_effect_corr = melt(r_effect_corr, id_vars='index')
r_effect_corr['distance'] = [r_distance.ix[i, c] for i, c in zip(*(r_effect_corr['index'], r_effect_corr['reaction']))]
r_effect_corr['distance'] = ['>7' if i > 7 or not np.isfinite(i) else str(i) for i in r_effect_corr['distance']]
r_effect_corr = r_effect_corr[[i != c for i, c in zip(*(r_effect_corr['index'], r_effect_corr['reaction']))]]
print '[INFO] Reactions effect size correlation calculated!'

g = sns.factorplot('distance', 'value', data=r_effect_corr, kind='box')
g.set_xlabels('Distance')
g.set_ylabels('Correlation')
plt.savefig(wd + 'reports/%s_reaction_activity_distance.pdf' % version, bbox_inches='tight')
plt.close('all')

#
e_k_corr = [(k, r, pearson(kinase_df.ix[k], r_effect.ix[r])) for r in r_effect.index for k in kinase_df.index]
e_k_corr = DataFrame([(k, r, c, p) for k, r, (c, p) in e_k_corr], columns=['kinase', 'reaction', 'cor', 'pvalue']).dropna()

kinases_to_keep = kinase_df[kinase_df.count(1) > 10].index
e_k_corr = e_k_corr[[k in kinases_to_keep for k in e_k_corr['kinase']]]

e_k_corr_df = pivot_table(e_k_corr, values='cor', index='reaction', columns='kinase')
print '[INFO] Reactions/Kinases correlations calculated'

#
e_k_corr_corr = e_k_corr_df.T.corr()
e_k_corr_corr['index'] = e_k_corr_corr.index
e_k_corr_corr = melt(e_k_corr_corr, id_vars='index')
e_k_corr_corr['distance'] = [r_distance.ix[i, c] for i, c in zip(*(e_k_corr_corr['index'], e_k_corr_corr['reaction']))]
e_k_corr_corr['distance'] = ['>7' if i > 7 or not np.isfinite(i) else str(i) for i in e_k_corr_corr['distance']]
e_k_corr_corr = e_k_corr_corr[[i != c for i, c in zip(*(e_k_corr_corr['index'], e_k_corr_corr['reaction']))]]
print '[INFO] Reactions effect size correlation calculated!'

g = sns.factorplot('distance', 'value', data=e_k_corr_corr, kind='box')
g.set_xlabels('Distance')
g.set_ylabels('Correlation')
plt.savefig(wd + 'reports/%s_reaction_activity_kinase_distance.pdf' % version, bbox_inches='tight')
plt.close('all')

# Import protein-protein interactions networks
dbs, gene_reactions = {}, dict(model.get_reactions_by_genes(model.get_genes()).items())
for bkg_type in ['string', 'phosphogrid']:
    if bkg_type == 'phosphogrid':
        db = read_csv(wd + 'files/phosphosites.txt', sep='\t').loc[:, ['KINASES_ORFS', 'ORF_NAME']]
        db = {(k, r['ORF_NAME']) for i, r in db.iterrows() for k in r['KINASES_ORFS'].split('|') if k != '-'}

    elif bkg_type == 'string':
        db = read_csv(wd + 'files/4932.protein.links.v9.1.txt', sep=' ')
        db_threshold = db['combined_score'].max() * 0.8
        db = db[db['combined_score'] > db_threshold]
        db = {(source.split('.')[1], target.split('.')[1]) for source, target in zip(db['protein1'], db['protein2'])}

    db = {(s, t) for s, t in db if s != t}
    print '[INFO] %s: %d' % (bkg_type, len(db))

    db = {(s, t) for s, t in db if t in gene_reactions}
    print '[INFO] %s, only enzyme targets: %d' % (bkg_type, len(db))

    db = {(s, r) for s, t in db for r in gene_reactions[t]}
    print '[INFO] %s, only enzymatic reactions: %d' % (bkg_type, len(db))

    db = {(k, r) for k, r in db if r in e_k_corr_df.index}
    print '[INFO] %s, only measured enzymatic reactions: %d' % (bkg_type, len(db))

    dbs[bkg_type] = db

print '[INFO] Kinase/Enzymes interactions data-bases imported'

# Enrichment for kinase/enzymes interactions
int_enrichment = []
for bkg_type, db in dbs.items():
    db_proteins = {c for i in db for c in i}

    M = {(k, r) for k, r in set(zip(e_k_corr['kinase'], e_k_corr['reaction'])) if k in db_proteins}
    n = M.intersection(db)

    thresholds, fractions, pvalues = np.arange(0, 3.0, .2), [], []
    N_pairs, x_pairs = [], []

    for threshold in thresholds:
        N = e_k_corr.loc[-np.log10(e_k_corr['pvalue']) > threshold]
        N = set(zip(N['kinase'], N['reaction'])).intersection(M)

        x = N.intersection(n)

        fraction = float(len(x)) / float(len(N)) if float(len(N)) != 0.0 else 0
        p_value = hypergeom.sf(len(x), len(M), len(n), len(N))

        fractions.append(fraction)
        pvalues.append(p_value)

        N_pairs.append(copy.copy(N))
        x_pairs.append(copy.copy(x))

        int_enrichment.append((bkg_type, threshold, fraction, p_value, len(M), len(n), len(N), len(x)))

        print 'Threshold: %.2f, M: %d, n: %d, N: %d, x: %d, fraction: %.4f, p-value: %.4f' % (threshold, len(M), len(n), len(N), len(x), fraction, p_value)

int_enrichment = DataFrame(int_enrichment, columns=['db', 'thres', 'fraction', 'pvalue', 'M', 'n', 'N', 'x'])

# Build information table
info_table = e_k_corr.copy()

info_table['kinase_count'] = [kinase_df.ix[k].count() for k in info_table['kinase']]

info_table['pvalue_log'] = [-np.log10(i) for i in info_table['pvalue']]
info_table['cor_abs'] = [np.abs(i) for i in info_table['cor']]

# Kinase/Enzyme interactions via metabolite correlations
for bkg_type, db in dbs.items():
    info_table['class_%s' % bkg_type] = [int(i in db) for i in zip(info_table['kinase'], info_table['reaction'])]

# Other metrics
info_table['euclidean'] = [metric(euclidean_distances, r_effect.ix[m, strains], kinase_df.ix[k, strains])[0][0] for k, m in zip(info_table['kinase'], info_table['reaction'])]

info_table['linear_kernel'] = [metric(linear_kernel, r_effect.ix[m, strains], kinase_df.ix[k, strains])[0][0] for k, m in zip(info_table['kinase'], info_table['reaction'])]
info_table['linear_kernel_abs'] = info_table['linear_kernel'].abs()

# ---- Kinase - enzymes interactions enrichment
(f, enrichemnt_plot), pos = plt.subplots(len(dbs), 3, figsize=(20, (6 * len(dbs)))), 0
for bkg_type in dbs:
    # Ploting
    ax = enrichemnt_plot[pos][0]
    N_thres = 2.0
    values = int_enrichment.loc[np.bitwise_and(int_enrichment['db'] == bkg_type, int_enrichment['thres'] == N_thres)]

    plot_df = values[['M', 'n', 'N', 'x']].T
    plot_df.columns = ['value']
    plot_df['variable'] = ['WO filter', 'WO filter', 'W filter', 'W filter']
    plot_df['type'] = ['all', 'reported', 'all', 'reported']

    sns.barplot('variable', 'value', 'type', data=plot_df, ci=0, x_order=['WO filter', 'W filter'], ax=ax)
    sns.despine(left=True, bottom=True, ax=ax)
    ax.set_ylabel(bkg_type.upper())
    ax.set_xlabel('')
    ax.set_title('M: %d, n: %d, N: %d, x: %d, p-value: %.4f\nFilter: -log10(cor p-value) > %.1f' % (values['M'], values['n'], values['N'], values['x'], values['pvalue'], N_thres))

    # Hypergeometric specific thresold analysis
    ax = enrichemnt_plot[pos][1]
    plot_df = int_enrichment.loc[int_enrichment['db'] == bkg_type, ['thres', 'fraction']].copy()
    sns.barplot('thres', 'fraction', data=plot_df, ax=ax)
    sns.despine(left=True, bottom=True, ax=ax)
    ax.set_xlabel('correlation threshold (-log10 p-value)')
    ax.set_ylabel('% of reported Kinase/Enzyme association')

    ax = enrichemnt_plot[pos][2]
    # ROC plot analysis
    for roc_metric in ['pvalue_log']:
        curve_fpr, curve_tpr, _ = roc_curve(info_table['class_%s' % bkg_type], info_table[roc_metric])
        curve_auc = auc(curve_fpr, curve_tpr)

        ax.plot(curve_fpr, curve_tpr, label='%s (area = %0.2f)' % (roc_metric, curve_auc))

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    sns.despine(left=True, bottom=True, ax=ax)
    ax.legend(loc='lower right')

    pos += 1

plt.savefig(wd + 'reports/%s_kinase_enzyme.pdf' % version, bbox_inches='tight')
plt.close('all')
print '[INFO] Plotting done: ', wd + 'plots/%s_kinase_enzyme.pdf' % version