import re
import copy
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from yeast_phospho import wd
from yeast_phospho.utils import pearson, metric
from sklearn.metrics.pairwise import euclidean_distances, linear_kernel
from pandas.tools.pivot import pivot_table
from bioservices.kegg import KEGGParser, KEGG
from pandas import DataFrame, read_csv
from pandas.core.index import Index
from scipy.stats.distributions import hypergeom
from pymist.reader.sbml_reader import read_sbml_model


sns.set_style('ticks')

# Import Kegg bioservice
kegg_srv, kegg_prs = KEGG(verbose=True, cache=True), KEGGParser()

# Import metabolites map
m_map = read_csv('%s/files/metabolite_mz_map_kegg.txt' % wd, sep='\t')
m_map['mz'] = Index([str(i) for i in m_map['mz']], dtype=str)
m_map = m_map.drop_duplicates('mz').drop_duplicates('formula')
m_map = m_map.groupby('mz')['name'].apply(lambda i: '; '.join(i)).to_dict()

# Import kinase activity
k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)

# Import metabolomics
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)
metabolomics.index = Index([str(i) for i in metabolomics.index], dtype=str)

# Overlapping kinases/phosphatases knockout
strains, metabolites, kinases = list(set(k_activity.columns).intersection(set(metabolomics.columns))), list(metabolomics.index), list(k_activity.index)
k_activity, metabolomics = k_activity[strains], metabolomics.loc[m_map.keys(), strains].dropna()


# ---- Correlate metabolic fold-changes with kinase activities
cor_df = [(m, k, pearson(metabolomics.loc[m, strains], k_activity.loc[k, strains])) for m in metabolomics.index for k in k_activity.index]
cor_df = DataFrame([(m, k, c, p, n) for m, k, (c, p, n) in cor_df], columns=['metabolite', 'kinase', 'cor', 'pvalue', 'n_meas'])
cor_df['adj.pvalue'] = multipletests(cor_df['pvalue'], method='fdr_bh')[1]
cor_df['metabolite_name'] = [m_map[i] if i in m_map else i for i in cor_df['metabolite']]
cor_df['kinase_name'] = [acc_name[i] if i in acc_name else i for i in cor_df['kinase']]
print '[INFO] Correaltion between metabolites and kinases done'


# ---- Calculate metabolite distances
# Import metabolic model mapping
model_met_map = read_csv(wd + 'files/metabolite_mz_map_dobson.txt', sep='\t', index_col='id')
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

# Gene metabolite association
met_reactions = {kegg_id: set(model.s[met_id].keys()) for met_id, kegg_id in model_met_map.items() if met_id in model.s}
gene_reactions = dict(model.get_reactions_by_genes(model.get_genes()).items())


# ---- Correlate metabolites with distances



# Read protein interactions dbs
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

    db = {(s, t) for s, t in db if t in gene_reactions}
    print '[INFO] %s, only enzyme targets: %d' % (bkg_type, len(db))

    db = {(s, r) for s, t in db for r in gene_reactions[t]}
    print '[INFO] %s, only enzymatic reactions: %d' % (bkg_type, len(db))

    s_matrix = model.get_stoichiometric_matrix()
    db = {(s, m) for s, r in db if r in s_matrix.columns for m in set((s_matrix[s_matrix[r] != 0]).index)}
    print '[INFO] %s, only enzymatic reactions metabolites: %d' % (bkg_type, len(db))

    db = {(s, model_met_map[met_id]) for s, met_id in db if met_id in model_met_map}
    print '[INFO] %s, only measured enzymatic reactions metabolites: %d' % (bkg_type, len(db))

    dbs[bkg_type] = db

print '[INFO] Kinase/Enzymes interactions data-bases imported'

# Filter metabolites within the model
model_met_map['kegg'] = [model_met_map[i] if i in model_met_map else np.NaN for i in metabolomics.index]
metabolomics = metabolomics.dropna(subset=['kegg']).groupby('kegg').first()
print '[INFO] [METABOLOMICS] (filtered metabolites within model/kegg): ', metabol_df.shape

# Build information table
info_table = [(k, m, cor_df.ix[m, k]) for k in cor_df.columns for m in cor_df.index if not np.isnan(cor_df.ix[m, k])]
info_table = DataFrame(info_table, columns=['kinase', 'metabolite', 'correlation', 'pvalue'])

info_table = info_table[[i in metabolites_map.index for i in info_table['metabolite']]]

info_table['metabolite_name'] = [metabolites_map.loc[m, 'id'] for m in info_table['metabolite']]
info_table['kinase_name'] = [acc_name.loc[k, 'gene'].split(';')[0] for k in info_table['kinase']]

info_table['metabolite_count'] = [metabol_df.ix[m].count() for m in info_table['metabolite']]
info_table['kinase_count'] = [kinase_df.ix[k].count() for k in info_table['kinase']]

info_table['pvalue_log'] = [-np.log10(i) for i in info_table['pvalue']]
info_table['correlation_abs'] = [np.abs(i) for i in info_table['correlation']]

# Kinase/Enzyme interactions via metabolite correlations
for bkg_type, db in dbs.items():
    info_table['class_%s' % bkg_type] = [int(i in db) for i in zip(info_table['kinase'], info_table['metabolite_name'])]

# Other metrics
info_table['euclidean'] = [metric(euclidean_distances, metabol_df.ix[m, strains], kinase_df.ix[k, strains])[0][0] for k, m in zip(info_table['kinase'], info_table['metabolite'])]

info_table['linear_kernel'] = [metric(linear_kernel, metabol_df.ix[m, strains], kinase_df.ix[k, strains])[0][0] for k, m in zip(info_table['kinase'], info_table['metabolite'])]
info_table['linear_kernel_abs'] = info_table['linear_kernel'].abs()

# Kinase/Enzyme enrichment
info_table = read_csv(wd + 'tables/information_table.tab', sep='\t')

int_enrichment = []
for bkg_type, db in dbs.items():
    db_proteins = {c for i in db for c in i}

    M = {(k, m) for k, m in set(zip(info_table['kinase'], info_table['metabolite_name'])) if k in db_proteins}
    n = M.intersection(db)

    thresholds, fractions, pvalues = np.arange(0, 3.0, .2), [], []
    N_pairs, x_pairs = [], []

    for threshold in thresholds:
        N = info_table.loc[info_table['pvalue_log'] > threshold]
        N = set(zip(N['kinase'], N['metabolite_name'])).intersection(M)

        x = N.intersection(n)

        fraction = float(len(x)) / float(len(N)) if float(len(N)) != 0.0 else 0
        p_value = hypergeom.sf(len(x), len(M), len(n), len(N))

        fractions.append(fraction)
        pvalues.append(p_value)

        N_pairs.append(copy.copy(N))
        x_pairs.append(copy.copy(x))

        print 'Threshold: %.2f, M: %d, n: %d, N: %d, x: %d, fraction: %.4f, p-value: %.4f' % (threshold, len(M), len(n), len(N), len(x), fraction, p_value)

        int_enrichment.append((bkg_type, threshold, fraction, p_value, len(M), len(n), len(N), len(x)))

int_enrichment = DataFrame(int_enrichment, columns=['db', 'thres', 'fraction', 'pvalue', 'M', 'n', 'N', 'x'])
print '[INFO] Kinase/Enzyme enrichment ready'
