import re
import copy
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from yeast_phospho import wd
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import correlation
from yeast_phospho.utils import pearson, metric, spearman
from sklearn.metrics.pairwise import euclidean_distances, linear_kernel, manhattan_distances, cosine_distances
from sklearn.metrics import roc_curve, auc
from pandas import DataFrame, read_csv, melt
from pandas.core.index import Index
from sklearn.linear_model import Lasso
from scipy.stats.distributions import hypergeom
from pymist.reader.sbml_reader import read_sbml_model


sns.set(style='ticks', palette='pastel', color_codes=True)


# Import id maps
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)

# Import metabolites map
m_map = read_csv('%s/files/metabolite_mz_map_kegg.txt' % wd, sep='\t')
m_map['mz'] = Index(['%.2f' % i for i in m_map['mz']], dtype=str)
m_map = m_map.drop_duplicates('mz').drop_duplicates('formula')
m_map = m_map.groupby('mz')['name'].apply(lambda i: '; '.join(i)).to_dict()


# ---- Calculate metabolite distances
# Import metabolic model mapping
model_met_map = read_csv(wd + 'files/metabolite_mz_map_dobson.txt', sep='\t', index_col='id')
model_met_map['mz'] = [float('%.2f' % i) for i in model_met_map['mz']]
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

# # Calculate metabolites x metabolites distance matrix
# _m_distance = s_matrix.dot(s_matrix.T).abs()
# _m_distance = (_m_distance != 0) + 0
#
# m_distance = DataFrame(nx.all_pairs_dijkstra_path_length(nx.from_numpy_matrix(_m_distance.values, create_using=nx.DiGraph())))
# m_distance.index = _m_distance.index
# m_distance.columns = _m_distance.columns
# print '[INFO] Metabolites distance calculated!'
#
# # Calculate reactions x reactions distance matrix
# _r_distance = s_matrix.T.dot(s_matrix).abs()
# _r_distance = (_r_distance != 0) + 0
#
# r_distance = DataFrame(nx.all_pairs_dijkstra_path_length(nx.from_numpy_matrix(_r_distance.values, create_using=nx.DiGraph())))
# r_distance.index = _r_distance.index
# r_distance.columns = _r_distance.columns
# print '[INFO] Metabolic reactions distance calculated!'

# Get reactions metabolites
r_metabolites = {r: set(s_matrix.ix[s_matrix[r] != 0, r].index) for r in s_matrix.columns}

# Gene metabolite association
met_reactions = {kegg_id: set(model.s[met_id].keys()) for met_id, kegg_id in model_met_map.items() if met_id in model.s}
gene_reactions = dict(model.get_reactions_by_genes(model.get_genes()).items())

# ---- Read protein interactions dbs
dbs = {}
for bkg_type in ['string', 'phosphogrid']:
    if bkg_type == 'phosphogrid':
        db = read_csv(wd + 'files/PhosphoGrid.txt', sep='\t').loc[:, ['KINASES_ORFS', 'ORF_NAME']]
        db = {(k, r['ORF_NAME']) for i, r in db.iterrows() for k in r['KINASES_ORFS'].split('|') if k != '-'}

    elif bkg_type == 'string':
        db = read_csv(wd + 'files/4932.protein.links.v9.1.txt', sep=' ')
        db_threshold = db['combined_score'].max() * 0.8
        db = db[db['combined_score'] > db_threshold]
        db = {(source.split('.')[1], target.split('.')[1]) for source, target in zip(db['protein1'], db['protein2'])}

    db = {(s, t) for s, t in db if s != t}
    print '[INFO] %s: %d' % (bkg_type, len(db))

    db = {(s, x) for s, t in db for x in [s, t] if x in gene_reactions}
    print '[INFO] %s, only enzyme targets: %d' % (bkg_type, len(db))

    db = {(s, r) for s, t in db for r in gene_reactions[t] if r in s_matrix.columns}
    print '[INFO] %s, only enzymatic reactions: %d' % (bkg_type, len(db))

    db = {(s, m) for s, r in db if r in s_matrix.columns for m in set((s_matrix[s_matrix[r] != 0]).index)}
    print '[INFO] %s, only enzymatic reactions metabolites: %d' % (bkg_type, len(db))

    db = {(s, model_met_map[m]) for s, m in db if m in model_met_map}
    print '[INFO] %s, only measured enzymatic reactions metabolites: %d' % (bkg_type, len(db))

    dbs[bkg_type] = db

db = dbs['string'].union(dbs['phosphogrid'])
print '[INFO] Kinase/Enzymes interactions data-bases imported'


# ---- Import data-sets
datasets_files = [
    ('%s/tables/kinase_activity_steady_state.tab' % wd, '%s/tables/metabolomics_steady_state.tab' % wd, 'no_growth'),
    ('%s/tables/kinase_activity_steady_state_with_growth.tab' % wd, '%s/tables/metabolomics_steady_state_growth_rate.tab' % wd, 'with_growth'),
    ('%s/tables/kinase_activity_dynamic.tab' % wd, '%s/tables/metabolomics_dynamic.tab' % wd, 'dynamic')
]

for k_file, m_file, growth in datasets_files:
    # Import kinase activity
    k_activity = read_csv(k_file, sep='\t', index_col=0).replace(np.NaN, 0.0)

    # Import metabolomics
    metabolomics = read_csv(m_file, sep='\t', index_col=0)
    metabolomics = metabolomics[metabolomics.std(1) > .4]
    metabolomics = metabolomics[(metabolomics.abs() > .8).sum(1) > 0]

    # Overlapping kinases/phosphatases knockout
    strains = list(set(k_activity.columns).intersection(set(metabolomics.columns)))
    k_activity, metabolomics = k_activity[strains], metabolomics[strains].dropna()

    # ---- Correlate metabolic fold-changes with kinase activities
    lm = Lasso(alpha=.01).fit(k_activity.T, metabolomics.T)

    info_table = DataFrame(lm.coef_, index=metabolomics.index, columns=k_activity.index)
    info_table['metabolite'] = info_table.index
    info_table = melt(info_table, id_vars='metabolite', value_name='coef', var_name='kinase')
    # info_table = info_table[info_table['coef'] != 0.0]

    info_table['abs_coef'] = [np.abs(i) for i in info_table['coef']]
    info_table['inv_abs_coef'] = [1 - np.abs(i) for i in info_table['coef']]

    info_table['kinase_name'] = [acc_name[i] if i in acc_name else i for i in info_table['kinase']]

    info_table['kinase_count'] = [k_activity.ix[k].count() for k in info_table['kinase']]

    # Kinase/Enzyme interactions via metabolite correlations
    info_table['TP'] = [int(i in db) for i in zip(info_table['kinase'], info_table['metabolite'])]

    # Other metrics
    info_table['euclidean'] = [metric(euclidean_distances, metabolomics.ix[m, strains], k_activity.ix[k, strains])[0][0] for k, m in zip(info_table['kinase'], info_table['metabolite'])]
    info_table['manhattan'] = [metric(manhattan_distances, metabolomics.ix[m, strains], k_activity.ix[k, strains])[0][0] for k, m in zip(info_table['kinase'], info_table['metabolite'])]

    info_table['linear_kernel'] = [metric(linear_kernel, metabolomics.ix[m, strains], k_activity.ix[k, strains])[0][0] for k, m in zip(info_table['kinase'], info_table['metabolite'])]
    info_table['linear_kernel_abs'] = info_table['linear_kernel'].abs()

    info_table = info_table.dropna()
    print '[INFO] Correaltion between metabolites and kinases done'

    # ---- Kinase/Enzyme enrichment
    int_enrichment = []

    db_proteins = {c for i in db for c in i}

    M = {(k, m) for k, m in set(zip(info_table['kinase'], info_table['metabolite'])) if k in db_proteins}
    n = M.intersection(db)

    thresholds, fractions, pvalues = np.arange(0, .45, .05), [], []
    N_pairs, x_pairs = [], []

    for threshold in thresholds:
        N = info_table.loc[info_table['abs_coef'] > threshold]
        N = set(zip(N['kinase'], N['metabolite'])).intersection(M)

        x = N.intersection(n)

        fraction = float(len(x)) / float(len(N)) if float(len(N)) != 0.0 else 0
        p_value = hypergeom.sf(len(x), len(M), len(n), len(N))

        fractions.append(fraction)
        pvalues.append(p_value)

        N_pairs.append(copy.copy(N))
        x_pairs.append(copy.copy(x))

        print 'Threshold: %.2f, M: %d, n: %d, N: %d, x: %d, fraction: %.4f, p-value: %.4f' % (threshold, len(M), len(n), len(N), len(x), fraction, p_value)

        int_enrichment.append((threshold, fraction, p_value, len(M), len(n), len(N), len(x)))

    int_enrichment = DataFrame(int_enrichment, columns=['thres', 'fraction', 'pvalue', 'M', 'n', 'N', 'x'])
    print '[INFO] Kinase/Enzyme enrichment ready'

    # Ploting
    (f, enrichemnt_plot) = plt.subplots(1, 3, figsize=(20, 6))

    ax = enrichemnt_plot[0]
    N_thres = 0.4
    values = int_enrichment.loc[int_enrichment['thres'] == N_thres]

    plot_df = values[['M', 'n', 'N', 'x']].T
    plot_df.columns = ['value']
    plot_df['variable'] = ['WO filter', 'WO filter', 'W filter', 'W filter']
    plot_df['type'] = ['all', 'reported', 'all', 'reported']

    sns.barplot('variable', 'value', 'type', data=plot_df, ci=0, x_order=['WO filter', 'W filter'], ax=ax, color='gray')
    sns.despine(left=True, bottom=True, ax=ax)
    ax.set_ylabel('PhosphoGrid + String')
    ax.set_xlabel('')
    ax.set_title('M: %d, n: %d, N: %d, x: %d, p-value: %.4f\nFilter: -log10(cor p-value) > %.1f' % (values['M'], values['n'], values['N'], values['x'], values['pvalue'], N_thres))

    # Hypergeometric specific thresold analysis
    ax = enrichemnt_plot[1]
    plot_df = int_enrichment[['thres', 'fraction']].copy()
    sns.barplot(x='thres', y='fraction', data=plot_df, ax=ax, lw=0, color='gray')
    sns.despine(left=True, bottom=True, ax=ax)
    ax.set_xlabel('correlation threshold (-log10 p-value)')
    ax.set_ylabel('% of reported Kinase/Enzyme association')

    # ROC plot analysis
    ax = enrichemnt_plot[2]
    for roc_metric in ['abs_coef', 'euclidean', 'linear_kernel_abs', 'manhattan']:
        curve_fpr, curve_tpr, _ = roc_curve(info_table['TP'], info_table[roc_metric])
        curve_auc = auc(curve_fpr, curve_tpr)

        ax.plot(curve_fpr, curve_tpr, label='%s (area = %0.2f)' % (roc_metric, curve_auc))

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')

    plt.savefig('%s/reports/kinase_enzyme_enrichment_metabolomics_lm_%s.pdf' % (wd, growth), bbox_inches='tight')
    plt.close('all')
    print '[INFO] Plotting done: %s/reports/kinase_enzyme_enrichment_metabolomics_%s.pdf' % (wd, growth)
