import re
import copy
import random
import numpy as np
import seaborn as sns
import igraph as igraph
import itertools as it
import matplotlib.pyplot as plt
from numpy.ma import masked_invalid as mask
from sklearn.cross_validation import KFold, LeaveOneOut
from sklearn.feature_selection.univariate_selection import f_classif, SelectKBest, chi2, f_regression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics.pairwise import euclidean_distances, linear_kernel, manhattan_distances, rbf_kernel, polynomial_kernel, pairwise_distances
from sklearn.metrics.metrics import roc_curve, auc, jaccard_similarity_score, r2_score
from pandas.tools.pivot import pivot_table
from enrichment.enzyme_activity import metabolite_distance
from pymist.reader import sbml_reader
from bioservices.kegg import KEGGParser, KEGG
from pandas import DataFrame, Series, read_csv
from pandas.core.index import Index
from scipy.stats import spearmanr, ttest_ind, linregress
from scipy.stats.mstats import pearsonr
from scipy.stats.distributions import hypergeom
from enrichment.gsea import gsea


def remove_nan(values):
    return values[~np.isnan(values)]


def get_site(protein, peptide):
    pep_start = protein.find(re.sub('\[.+\]', '', peptide))
    pep_site_strat = peptide.find('[')
    site_pos = pep_start + pep_site_strat
    return protein[site_pos - 1] + str(site_pos)

def metric(func, x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    return func(x[mask], y[mask]) if np.sum(mask) > 5 else np.NaN


def my_pairwise(x, y, metric):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    return pairwise_distances(x[mask], y[mask], metric=metric) if np.sum(mask) > 5 else np.NaN


wd = '/Users/emanuel/Projects/projects/pymist/resources/yeast_phospho/'

# Seaborn configurations
sns.set_style('white')

# Import Kegg bioservice
kegg_srv, kegg_prs = KEGG(verbose=True, cache=True), KEGGParser()

# Import acc map to name form uniprot
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)

# Import growth rates
growth = read_csv(wd + 'files/strain_relative_growth_rate.txt', sep='\t', index_col=0)

# Import Phosphogrid network
network = read_csv(wd + 'files/phosphosites.txt', sep='\t').loc[:, ['ORF_NAME', 'PHOSPHO_SITE', 'KINASES_ORFS', 'PHOSPHATASES_ORFS', 'SEQUENCE']]
print '[INFO] [PHOSPHOGRID] ', network.shape

# Import phosphoproteomics
phospho_df = read_csv(wd + 'phosphoproteomics/allEvents.tsv', sep='\t')
phospho_df = phospho_df.pivot_table(values='logFC', index=['peptide', 'target'], columns='regulator', aggfunc=np.median)
print '[INFO] [PHOSPHOPROTEOMICS] merge repeated phosphopeptides, i.e. median : ', phospho_df.shape

# Import metabolites mass charge to metabolic model
metabolites_map = read_csv(wd + 'metabolomics/metabolite_mz_map_kegg.txt', sep='\t')  # _adducts
metabolites_map['mz'] = [str(i) for i in metabolites_map['mz']]
metabolites_map = metabolites_map.drop_duplicates('mz').drop_duplicates('formula')
metabolites_map = metabolites_map[metabolites_map['mod'] == '-H(+)']
metabolites_map.index = Index(metabolites_map['mz'], dtype=np.str)
metabolites_map_dict = metabolites_map['id'].to_dict()

metabolites_map.to_csv(wd + 'tables/metabolites_map.tab', sep='\t')
print '[INFO] Metabolites map filtered'

# Import metabolomics
metabol_df = read_csv(wd + 'metabolomics/Table_S3.txt', sep='\t')
metabol_df.index = [str(i) for i in metabol_df['m/z']]
metabol_df = metabol_df.drop('m/z', 1).dropna()
metabol_df = metabol_df[(metabol_df.abs() > 1.0).sum(1) > 1]
print '[INFO] [METABOLOMICS]: ', metabol_df.shape

# Filter metabolites within the model
metabol_df['kegg'] = [metabolites_map_dict[i] if i in metabolites_map_dict else np.NaN for i in metabol_df.index]
metabol_df = metabol_df.dropna(subset=['kegg']).groupby('kegg').first()
print '[INFO] [METABOLOMICS] (filtered metabolites within model/kegg): ', metabol_df.shape

# Overlapping kinases/phosphatases knockout
strains = list(set(phospho_df.columns).intersection(set(metabol_df.columns)))
print '[INFO] Overlaping conditions: ', len(strains), ' : ', strains

# Consider phoshopeptides with only one phosphosite
phospho_df = phospho_df.loc[[len(re.findall('\[[0-9]*\.?[0-9]*\]', peptide)) == 1 for peptide in phospho_df.index.levels[0]]]
print '[INFO] [PHOSPHOPROTEOMICS] (filtered phosphopetides with multiple phosphosites): ', phospho_df.shape

# Remove K and R aminoacids from the peptide head
phospho_df.index = phospho_df.index.set_levels([re.split('^[K|R]\.', x)[1] for x in phospho_df.index.levels[0]], 'peptide')

# Match peptide sequences to protein sequence and calculate peptide phosphorylation site
pep_match = {peptide: set(network.loc[network['ORF_NAME'] == target, 'SEQUENCE']) for (peptide, target), r in phospho_df.iterrows()}
pep_site = {peptide: target + '_' + get_site(list(pep_match[peptide])[0].upper(), peptide) for (peptide, target), r in phospho_df.iterrows() if len(pep_match[peptide]) == 1}

# Merge phosphosites with median
phospho_df['site'] = [pep_site[peptide] if peptide in pep_site else np.NaN for (peptide, target), r in phospho_df.iterrows()]
phospho_df = phospho_df.groupby('site').median()
print '[INFO] [PHOSPHOPROTEOMICS] (merge phosphosites, i.e median): ', phospho_df.shape

# Sort data-sets in same strain order and export data-sets
metabol_df, phospho_df = metabol_df[strains], phospho_df[strains]
metabol_df.to_csv(wd + 'tables/metabolomics.tab', sep='\t')
phospho_df.to_csv(wd + 'tables/phosphoproteomics.tab', sep='\t')
print '[INFO] Data-sets exported'

# Filter interactions without kinases/phosphatases
network = network.loc[np.bitwise_or(network['KINASES_ORFS'] != '-', network['PHOSPHATASES_ORFS'] != '-')]
print '[INFO] [PHOSPHOGRID] (filter non-kinase/phosphatase interactions): ', network.shape

# Split into multiple interactions into different lines and remove self-phosphorylation events
network['SOURCE'] = network['KINASES_ORFS'] + '|' + network['PHOSPHATASES_ORFS']
network = [(k, r['ORF_NAME'] + '_' + r['PHOSPHO_SITE']) for i, r in network.iterrows() for k in r['SOURCE'].split('|') if k != '-' and r['ORF_NAME'] != k]
network = DataFrame(network, columns=['SOURCE', 'TARGET'])
print '[INFO] [PHOSPHOGRID] (split into multiple interactions into different lines and remove self-phosphorylation events): ', network.shape

# Kinases enrichment
kinases = set(network.loc[network['SOURCE'] != '', 'SOURCE'])
kinases_targets = {k: set(network.loc[network['SOURCE'] == k, 'TARGET']) for k in kinases}
kinases_targets = {k: t.intersection(phospho_df.index) for k, t in kinases_targets.items() if len(t.intersection(phospho_df.index)) > 2}
# kinase_df = [(k, ko, gsea(phospho_df[ko], targets, True, 1000)[:2]) for k, targets in kinases_targets.items() for ko in strains]
# kinase_df = [(k, ko, -np.log10(pvalue) if es < 0 else np.log10(pvalue)) for k, ko, (es, pvalue) in kinase_df]
# kinase_df = DataFrame(kinase_df, columns=['kinase', 'strain', 'score']).dropna()
# kinase_df = pivot_table(kinase_df, values='score', index='kinase', columns='strain')
# print '[INFO] GSEA for kinase enrichment done. ', kinase_df.shape
#
# # Export matrix
# kinase_df.to_csv(wd + 'tables/kinase_enrichment_df.tab', sep='\t')
# print '[INFO] Kinase enrichment matrix exported to: ', wd + 'files/kinase_enrichment_df.tab'

# Import matrix
kinase_df = read_csv(wd + 'tables/kinase_enrichment.tab', sep='\t', index_col=0)
kinase_df = kinase_df[kinase_df.count(1) > 110]  # Threshold: number of measurements for kinases enrichments

# Metabolic model distance
model = sbml_reader.read_sbml_model('/Users/emanuel/Projects/resources/metabolic_models/1752-0509-4-145-s1/yeast_4.04.xml')
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

model_met_map = read_csv(wd + 'metabolomics/metabolite_mz_map_dobson.txt', sep='\t', index_col='id')
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


# Metabolomics phospho correlation
def randomise_matrix(matrix):
    matrix_copy = matrix.copy()
    movers = ~np.isnan(matrix_copy.values)
    matrix_copy.values[movers] = np.random.permutation(matrix_copy.values[movers])
    return matrix_copy

n_permutations = 1000
rand_kinases = {i: randomise_matrix(kinase_df) for i in range(n_permutations)}
print '[INFO] Random kinases matrices generated'

rand_cor_dist = {k: {m: [r2_score(mask(metabol_df.loc[m, strains]), mask(rand_kinase.loc[k, strains])) for i, rand_kinase in rand_kinases.items()] for m in metabol_df.index} for k in kinase_df.index}
print '[INFO] Random correlation distributions generated'


def empirical_pvalue(m, k, cor):
    random_cor = rand_cor_dist[k][m]
    count = np.sum(random_cor >= cor) if cor > 0 else np.sum(random_cor <= cor)
    return 1.0 / n_permutations if count == 0 else float(count) / n_permutations

cor_df = [(m, k, pearsonr(mask(metabol_df.loc[m, strains]), mask(kinase_df.loc[k, strains]))[0]) for m in metabol_df.index for k in kinase_df.index]
cor_df = DataFrame([(m, k, c, empirical_pvalue(m, k, c)) for m, k, c in cor_df], columns=['metabolite', 'kinase', 'cor', 'pval'])
cor_df_pvalue = pivot_table(cor_df, values='pval', index='metabolite', columns='kinase')
cor_df = pivot_table(cor_df, values='cor', index='metabolite', columns='kinase')
cor_df.to_csv(wd + 'tables/met_kin_correlation.tab', sep='\t')

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

# Build information table
info_table = [(k, m, cor_df.ix[m, k], cor_df_pvalue.ix[m, k]) for k in cor_df.columns for m in cor_df.index if not np.isnan(cor_df.ix[m, k]) and m in met_reactions]
info_table = DataFrame(info_table, columns=['kinase', 'metabolite', 'correlation', 'pvalue'])

info_table['metabolite_name'] = [metabolites_map.ix[metabolites_map['id'] == m, 'name'][0] for m in info_table['metabolite']]
info_table['kinase_name'] = [acc_name.loc[k, 'gene'].split(';')[0] for k in info_table['kinase']]

info_table['metabolite_count'] = [metabol_df.ix[m].count() for m in info_table['metabolite']]
info_table['kinase_count'] = [kinase_df.ix[k].count() for k in info_table['kinase']]

info_table['pvalue_log'] = [-np.log10(i) for i in info_table['pvalue']]
info_table['correlation_abs'] = [np.abs(i) for i in info_table['correlation']]

# Kinase/Enzyme interactions via metabolite correlations
for bkg_type, db in dbs.items():
    info_table['class_%s' % bkg_type] = [int(i in db) for i in zip(info_table['kinase'], info_table['metabolite'])]

# Other metrics
info_table['euclidean'] = [metric(euclidean_distances, metabol_df.ix[m], kinase_df.ix[k])[0][0] for k, m in zip(info_table['kinase'], info_table['metabolite'])]

info_table['linear_kernel'] = [metric(linear_kernel, metabol_df.ix[m], kinase_df.ix[k])[0][0] for k, m in zip(info_table['kinase'], info_table['metabolite'])]
info_table['linear_kernel_abs'] = info_table['linear_kernel'].abs()

info_table['manhattan_distances'] = [metric(manhattan_distances, metabol_df.ix[m], kinase_df.ix[k])[0][0] for k, m in zip(info_table['kinase'], info_table['metabolite'])]

info_table['rbf_kernel'] = [metric(rbf_kernel, metabol_df.ix[m], kinase_df.ix[k])[0][0] for k, m in zip(info_table['kinase'], info_table['metabolite'])]

info_table['polynomial_kernel'] = [metric(polynomial_kernel, metabol_df.ix[m], kinase_df.ix[k])[0][0] for k, m in zip(info_table['kinase'], info_table['metabolite'])]

info_table['cosine'] = [my_pairwise(metabol_df.ix[m], kinase_df.ix[k], 'cosine')[0][0] for k, m in zip(info_table['kinase'], info_table['metabolite'])]

# Export information table
info_table = info_table.sort('pvalue')
info_table.to_csv(wd + 'tables/information_table.tab', sep='\t', index=False)
print '[INFO] Information table ready'

# Kinase/Enzyme enrichment
int_enrichment = []
for bkg_type, db in dbs.items():
    db_proteins = {c for i in db for c in i}

    M = {(k, m) for k, m in set(zip(info_table['kinase'], info_table['metabolite'])) if k in db_proteins}
    n = M.intersection(db)

    thresholds, fractions, pvalues = np.arange(0, 3.0, .2), [], []
    N_pairs, x_pairs = [], []

    for threshold in thresholds:
        N = info_table.loc[info_table['pvalue_log'] > threshold]
        N = set(zip(N['kinase'], N['metabolite'])).intersection(M)

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

# Machine learning: metabolites -> growth
X, Y = kinase_df.copy(), metabol_df.copy()

Xs, Ys = Y.copy().T.values, growth.loc[strains, 'relative_growth'].copy().values / 100

scores = []
for train, test in LeaveOneOut(Ys.shape[0]):
    lm = Lasso().fit(Xs[train], Ys[train])
    score = np.linalg.norm(lm.predict(Xs[test]) - Ys[test])
    scores.append((score, strains[test[0]]))
print '[INFO] Model training done'

scores = DataFrame(scores, columns=['error', 'strain']).sort(columns='error')
scores['strain_name'] = [acc_name.loc[x, 'gene'].split(';')[0] for x in scores['strain']]

scores.to_csv(wd + 'tables/lm_growth_prediction.tab', index=False, sep='\t')
print '[INFO] Growth prediction done!'

# Kinases -> metabolites
pred_df, meas_df, error_df, m_features = [], [], [], dict()
models = {}
for i in range(Y.shape[0]):
    Xs, Ys = X.T.copy().replace(np.NaN, 0.0), Y.ix[i].copy()

    samples, features, metabolite = Xs.index, Xs.columns, Y.index[i]

    m_features[metabolite] = []
    models[metabolite] = {}

    cv = LeaveOneOut(len(samples))
    for train, test in cv:
        train_i, test_i = samples[train], samples[test]

        fs = SelectKBest(f_regression, 10).fit(Xs.ix[train_i], Ys[train_i])
        m_features[metabolite].extend(features[fs.get_support()])

        # lm_1 = SVR(kernel='linear').fit(Xs.ix[train_i], Ys[train_i])
        # lm_2 = RidgeCV().fit(Xs.ix[train_i], Ys[train_i])
        lm = LinearRegression(normalize=True).fit(Xs.ix[train_i], Ys[train_i])

        pred = lm.predict(Xs.ix[test_i])[0]

        meas = Ys[test_i][0]
        error = np.linalg.norm(pred - meas)

        pred_df.append((metabolite, test_i[0], pred))
        meas_df.append((metabolite, test_i[0], meas))
        error_df.append((metabolite, test_i[0], error))

        models[metabolite][test_i[0]] = lm

        print metabolite, test_i[0], pred, meas, error

error_df = pivot_table(DataFrame(error_df, columns=['metabolite', 'strain', 'value']), values='value', index='strain', columns='metabolite')
meas_df = pivot_table(DataFrame(meas_df, columns=['metabolite', 'strain', 'value']), values='value', index='strain', columns='metabolite')
pred_df = pivot_table(DataFrame(pred_df, columns=['metabolite', 'strain', 'value']), values='value', index='strain', columns='metabolite')

m_features_df = {k: dict(zip(*np.unique(v, return_counts=True))) for k, v in m_features.items()}
m_features_df = DataFrame(m_features_df).replace(np.NaN, 0.0)

error_df.to_csv(wd + 'tables/lm_error.tab', sep='\t')
meas_df.to_csv(wd + 'tables/lm_measured.tab', sep='\t')
pred_df.to_csv(wd + 'tables/lm_predicted.tab', sep='\t')
m_features_df.to_csv(wd + 'tables/lm_features.tab', sep='\t')
print '[INFO] Model training done'

# # Test case
# kinases_targets_test = {k: t.intersection(phospho_df.index) for k, t in kinases_targets.items() if len(t.intersection(phospho_test_df.index)) > 2}
# kinase_test_df = [(k, gsea(phospho_test_df.to_dict()['fc'], targets, True, 1000)[:2]) for k, targets in kinases_targets_test.items()]
# kinase_test_df = Series({k: -np.log10(pvalue) if es < 0 else np.log10(pvalue) for k, (es, pvalue) in kinase_test_df})
#
# for m in set(metabol_test_df.index).intersection(metabol_df.index):
#     Xs, Ys = X.T.copy().replace(np.NaN, 0.0), Y.ix[m].copy()
#
#     lm = LinearRegression(normalize=True).fit(Xs, Ys)
#
#     Xs_test, Ys_test = [kinase_test_df[k] if k in kinase_test_df else 0.0 for k in kinase_df.index], metabol_test_df[m]
#     print lm.predict(Xs_test), Ys_test