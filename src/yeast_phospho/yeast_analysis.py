import re
import copy
import scipy
import numpy as np
import igraph as igraph
import itertools as it
from yeast_phospho import wd
from numpy.ma import masked_invalid as mask
from sklearn.metrics.pairwise import euclidean_distances, linear_kernel, manhattan_distances, rbf_kernel, polynomial_kernel, pairwise_distances
from sklearn.metrics import roc_curve, auc, jaccard_similarity_score, r2_score
from pandas.tools.pivot import pivot_table
from bioservices.kegg import KEGGParser, KEGG
from pandas import DataFrame, Series, read_csv
from pandas.core.index import Index
from scipy.stats.mstats import pearsonr
from scipy.stats.distributions import hypergeom
from pymist.enrichment.enzyme_activity import metabolite_distance
from pymist.reader.sbml_reader import read_sbml_model


def pearson(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = scipy.stats.pearsonr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue, x[mask], y[mask], np.sum(mask)


def remove_nan(values):
    return values[~np.isnan(values)]


def metric(func, x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    res = func(x[mask], y[mask]) if np.sum(mask) > 5 else np.NaN
    return res if np.isfinite(res) else [[res]]


def my_pairwise(x, y, metric):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    res = pairwise_distances(x[mask], y[mask], metric=metric) if np.sum(mask) > 5 else np.NaN
    return res if np.isfinite(res) else [[res]]


# Import Kegg bioservice
kegg_srv, kegg_prs = KEGG(verbose=True, cache=True), KEGGParser()

# Import acc map to name form uniprot
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)

# Import phospho log2 FC
phospho_df = read_csv(wd + 'tables/steady_state_phosphoproteomics.tab', sep='\t', index_col='site')

# Import metabol log2 FC
metabol_df = read_csv(wd + 'tables/steady_state_metabolomics.tab', sep='\t', index_col=0)
metabol_df.index = Index(metabol_df.index, dtype=str)

# Import kinase enrichment
kinase_df = read_csv(wd + 'tables/kinase_enrichment_df.tab', sep='\t', index_col=0)
# kinase_df = kinase_df[kinase_df.count(1) > 110]  # Threshold: number of measurements for kinases enrichments

# Import metabolites mass charge to metabolic model
metabolites_map = read_csv(wd + 'files/metabolomics/metabolite_mz_map_kegg.txt', sep='\t')  # _adducts
metabolites_map['mz'] = [str(i) for i in metabolites_map['mz']]
metabolites_map = metabolites_map.drop_duplicates('mz').drop_duplicates('formula')
metabolites_map = metabolites_map[metabolites_map['mod'] == '-H(+)']
metabolites_map.index = Index(metabolites_map['mz'], dtype=np.str)
metabolites_map_dict = metabolites_map['id'].to_dict()

metabolites_map.to_csv(wd + 'tables/metabolites_map.tab', sep='\t')
print '[INFO] Metabolites map filtered'

# Overlapping kinases/phosphatases knockout
strains = list(set(phospho_df.columns).intersection(set(metabol_df.columns)))
print '[INFO] Overlaping conditions: ', len(strains), ' : ', strains

# Sort data-sets in same strain order and export data-sets
metabol_df, phospho_df = metabol_df[strains], phospho_df[strains]

# Metabolomics phospho correlation
cor_df = [(m, k, pearson(metabol_df.loc[m, strains], kinase_df.loc[k, strains])[:2]) for m in metabol_df.index for k in kinase_df.index]
cor_df = DataFrame([(m, k, c, p) for m, k, (c, p) in cor_df], columns=['metabolite', 'kinase', 'cor', 'pval'])
cor_df_pvalue = pivot_table(cor_df, values='pval', index='metabolite', columns='kinase')
cor_df = pivot_table(cor_df, values='cor', index='metabolite', columns='kinase')
cor_df.to_csv(wd + 'tables/met_kin_correlation.tab', sep='\t')
print '[INFO] Correaltion between metabolites and kinases done'

# Metabolic model distance
model = read_sbml_model('/Users/emanuel/Projects/resources/metabolic_models/1752-0509-4-145-s1/yeast_4.04.xml')
model.remove_b_metabolites()
model.remove_metabolites({k for k, v in model.metabolites.items() if re.match('.* \[extracellular\]', v)})
model.remove_reactions(model.get_exchanges(True))
model.remove_reactions(['r_1812'])
model.remove_orphan_metabolites()
model.remove_not_used_reactions()

met_to_remove = [
    'acetyl-CoA', 'carbon dioxide', 'coenzyme A', 'L-glutamate', 'water', 'hydrogen peroxide',
    'H+', 'NAD(+)', 'NADH', 'NADP(+)', 'NADPH', 'ammonium', 'oxygen', 'phosphate', 'diphosphate', '2-oxoglutarate',
    'acyl-CoA', 'ADP', 'AMP', 'ATP', 'UDP', 'UMP', 'UTP', 'CDP', 'CMP', 'CTP', 'GDP', 'GMP', 'GTP'
]
met_to_remove = {k for m in met_to_remove for k, v in model.metabolites.items() if re.match('%s \[.*\]' % re.escape(m), v)}

model.remove_metabolites(met_to_remove)
model.remove_orphan_metabolites()
model.remove_not_used_reactions()

s_distance = metabolite_distance(model, drop_metabolites=[])

model_met_map = read_csv(wd + 'files/metabolomics/metabolite_mz_map_dobson.txt', sep='\t', index_col='id')
model_met_map['mz'] = [str(i) for i in model_met_map['mz']]
model_met_map = model_met_map.drop_duplicates('mz')['mz'].to_dict()
model_met_map = {k: metabolites_map_dict[str(v)] for k, v in model_met_map.items() if str(v) in metabolites_map_dict}

s_distance['kegg'] = [model_met_map[i] if i in model_met_map else i for i in s_distance.index]
s_distance = s_distance.groupby('kegg').min().T

s_distance['kegg'] = [model_met_map[i] if i in model_met_map else i for i in s_distance.index]
s_distance = s_distance.groupby('kegg').min()

s_distance = s_distance.ix[[i.startswith('C') for i in s_distance.index], [i.startswith('C') for i in s_distance.columns]]
s_distance.to_csv(wd + 'tables/metabolites_distances.tab', sep='\t')
print '[INFO] Metabolites distance matrix calculated'

# Gene metabolite association
met_reactions = {kegg_id: set(model.s[met_id].keys()) for met_id, kegg_id in model_met_map.items() if met_id in model.s}
gene_reactions = dict(model.get_reactions_by_genes(model.get_genes()).items())

# Read protein interactions dbs
dbs = {}
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

    s_matrix = model.get_stoichiometric_matrix()
    db = {(s, m) for s, r in db if r in s_matrix.columns for m in set((s_matrix[s_matrix[r] != 0]).index)}
    print '[INFO] %s, only enzymatic reactions metabolites: %d' % (bkg_type, len(db))

    db = {(s, model_met_map[met_id]) for s, met_id in db if met_id in model_met_map}
    print '[INFO] %s, only measured enzymatic reactions metabolites: %d' % (bkg_type, len(db))

    DataFrame(list(db), columns=['source', 'target']).to_csv(wd + 'tables/db_%s.tab' % bkg_type, sep='\t', index=False)
    dbs[bkg_type] = db

print '[INFO] Kinase/Enzymes interactions data-bases imported'

# Filter metabolites within the model
metabol_df['kegg'] = [metabolites_map_dict[i] if i in metabolites_map_dict else np.NaN for i in metabol_df.index]
metabol_df = metabol_df.dropna(subset=['kegg']).groupby('kegg').first()
print '[INFO] [METABOLOMICS] (filtered metabolites within model/kegg): ', metabol_df.shape

# Build information table
info_table = [(k, m, cor_df.ix[m, k], cor_df_pvalue.ix[m, k]) for k in cor_df.columns for m in cor_df.index if not np.isnan(cor_df.ix[m, k])]
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

# Export information table
info_table = info_table.dropna().sort('pvalue')
info_table.to_csv(wd + 'tables/information_table.tab', sep='\t', index=False)
print '[INFO] Information table ready'

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
int_enrichment.to_csv(wd + 'tables/interactions_enrichment.tab', sep='\t', index=False)
print '[INFO] Kinase/Enzyme enrichment ready'