import re
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tools as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pandas import DataFrame, read_csv, melt, pivot_table
from yeast_phospho import wd
from sklearn.metrics import roc_curve, auc
from yeast_phospho.utilities import get_metabolites_model_annot, metric, pearson, get_proteins_name
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, linear_kernel
from pymist.reader.sbml_reader import read_sbml_model
from scipy.stats.distributions import hypergeom
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression


# ---- Calculate metabolite distances
# Import metabolic model mapping
met_annot = get_metabolites_model_annot()

met_name = met_annot['Name'].to_dict()
met_2_id = {v: k for k, v in met_annot['ABBR'].to_dict().items()}


# Import metabolic model
model = read_sbml_model('/Users/emanuel/Projects/resources/metabolic_models/iMM904.v1.xml')


# Remove extracellular metabolites
s_matrix = model.get_stoichiometric_matrix()
s_matrix = s_matrix[[not i.endswith('_b') for i in s_matrix.index]]

# Remove biomass reactions
s_matrix = s_matrix.drop('R_biomass_SC5_notrace', axis=1)

# Remove highly connected metabolites
s_matrix = s_matrix.drop([i for i in s_matrix.index if i[2:-2] in ['nadph', 'coa', 'accoa', 'atp', 'ctp', 'udp']])

# Remove exchange and biomass reactions
reactions_to_remove = model.get_exchanges(True)
s_matrix = s_matrix.loc[:, [r not in reactions_to_remove for r in s_matrix.columns]]

# Remove un-mapped metabolites
s_matrix = s_matrix.ix[[i[2:-2] in set(met_annot['ABBR']) for i in s_matrix.index]]

# Get reactions products and substrates
r_substrates = {r: set(s_matrix[s_matrix[r] < 0].index) for r in s_matrix.columns}
r_products = {r: set(s_matrix[s_matrix[r] > 0].index) for r in s_matrix.columns}

# Swap stoichiometric values with ones
s_matrix = (s_matrix != 0) + 0

# Remove un-used metabolites and reactions
s_matrix = s_matrix.loc[:, s_matrix.sum() != 0]
s_matrix = s_matrix[s_matrix.sum(1) != 0]

# Get reactions metabolites
r_metabolites = {r: set(s_matrix.ix[s_matrix[r] != 0, r].index) for r in s_matrix.columns}


# Gene metabolite association
gene_reactions = dict(model.get_reactions_by_genes(model.get_genes()).items())
model_2_ion = {i: met_2_id[i[2:-2]] for i in s_matrix.index}


# ---- Read protein interactions dbs
dbs = {}
for bkg_type in ['string']:
    if bkg_type == 'phosphogrid':
        db = read_csv(wd + 'files/PhosphoGrid.txt', sep='\t')[['KINASES_ORFS', 'ORF_NAME']]
        db = {(k, r['ORF_NAME']) for i, r in db.iterrows() for k in r['KINASES_ORFS'].split('|') if k != '-'}

    elif bkg_type == 'biogrid':
        db = read_csv(wd + 'files/BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-3.4.127.tab', sep='\t', skiprows=35)[['INTERACTOR_A', 'INTERACTOR_B']]
        db = {(s, t) for s, t in db.values}

    elif bkg_type == 'string':
        db = read_csv(wd + 'files/4932.protein.links.v9.1.txt', sep=' ')

        thres = 650
        db = db[db['combined_score'] > thres]

        db = {(source.split('.')[1], target.split('.')[1]) for source, target in db[['protein1', 'protein2']].values}

        print 'thres: %.2f' % thres

    db = {(s, t) for s, t in db if s != t}
    print '[INFO] %s: %d' % (bkg_type, len(db))

    db = {(s, x) for s, t in db for x in [s, t] if x in gene_reactions}
    print '[INFO] %s, only enzyme targets: %d' % (bkg_type, len(db))

    db = {(s, r) for s, t in db for r in gene_reactions[t] if r in s_matrix.columns}
    print '[INFO] %s, only enzymatic reactions: %d' % (bkg_type, len(db))

    db = {(s, m) for s, r in db if r in s_matrix.columns for m in r_metabolites[r]}
    print '[INFO] %s, only enzymatic reactions metabolites: %d' % (bkg_type, len(db))

    db = {(s, model_2_ion[m]) for s, m in db if m in model_2_ion}
    print '[INFO] %s, only measured enzymatic reactions metabolites: %d' % (bkg_type, len(db))

    dbs[bkg_type] = db

db = dbs['string']
print '[INFO] Kinase/Enzymes interactions data-bases imported'


# ---- Protein - Metabolite connectivity map
cmap = DataFrame(list(db), columns=['kinase', 'metabolite'])
cmap['value'] = 1
cmap = pivot_table(cmap, values='value', index='kinase', columns='metabolite', fill_value=0)


# ---- Import metabolites map
m_map = read_csv('%s/files/metabolite_mz_map_kegg.txt' % wd, sep='\t')
m_map['mz'] = [float('%.2f' % i) for i in m_map['mz']]
m_map = m_map.drop_duplicates('mz').drop_duplicates('formula')
m_map = m_map.groupby('mz')['name'].apply(lambda i: '; '.join(i)).to_dict()


# ---- Import YORF names
acc_name = get_proteins_name()


# ---- Import
# Steady-state with growth
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)
metabolomics = metabolomics[metabolomics.std(1) > .4]
metabolomics.index = [str(i) for i in metabolomics.index]

k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)
k_activity = k_activity[(k_activity.count(1) / k_activity.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0)


# Steady-state without growth
metabolomics_ng = read_csv('%s/tables/metabolomics_steady_state_no_growth.tab' % wd, sep='\t', index_col=0)
metabolomics_ng = metabolomics_ng[metabolomics_ng.std(1) > .4]

k_activity_ng = read_csv('%s/tables/kinase_activity_steady_state_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_ng = k_activity_ng[(k_activity_ng.count(1) / k_activity_ng.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_ng = read_csv('%s/tables/tf_activity_steady_state_no_growth.tab' % wd, sep='\t', index_col=0)


# Dynamic
metabolomics_dyn = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0)
metabolomics_dyn = metabolomics_dyn[metabolomics_dyn.std(1) > .4]
metabolomics_dyn.index = [str(i) for i in metabolomics_dyn.index]

k_activity_dyn = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0)
k_activity_dyn = k_activity_dyn[(k_activity_dyn.count(1) / k_activity_dyn.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_dyn = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0)


# ---- Comparisons
comparisons = [
    (k_activity, metabolomics, 'Kinases steady-state', 15),
    (tf_activity, metabolomics, 'TFs steady-state', 15),

    (k_activity_ng, metabolomics_ng, 'Kinases steady-state (no growth)', 15),
    (tf_activity_ng, metabolomics_ng, 'TFs steady-state (no growth)', 15),

    (k_activity_dyn, metabolomics_dyn, 'Kinases dynamic', 10),
    (tf_activity_dyn, metabolomics_dyn, 'TFs dynamic', 10)
]


# ---- Perform kinase/enzyme enrichment
sns.set(style='ticks', palette='pastel', color_codes=True, context='paper')
(f, plot), pos = plt.subplots(len(comparisons), 2, figsize=(7, 4. * len(comparisons))), 0
for xss, yss, descprition, fs_k in comparisons:

    # Overlapping conditions
    conditions = list(set(xss.columns).intersection(set(yss.columns)))
    xs, ys = xss[conditions].T, yss[conditions].T

    # ---- Correlate metabolic fold-changes with kinase activities
    lm_models = {m: sm.OLS(ys[m], st.add_constant(xs)).fit_regularized(L1_wt=0) for m in ys.columns}

    info_table = [(m, f, c) for m in lm_models for f, c in lm_models[m].params.to_dict().items() if f != 'const' and c != 0 and np.isfinite(c)]
    info_table = DataFrame(info_table, columns=['metabolite', 'feature', 'coef'])

    info_table['score'] = info_table['coef'].abs().max() - info_table['coef'].abs()

    info_table['metabolite_name'] = [m_map[m] if m in m_map else str(m) for m in info_table['metabolite']]
    info_table['feature_name'] = [acc_name[k] if k in acc_name else str(k) for k in info_table['feature']]

    # Kinase/Enzyme interactions via metabolite correlations
    info_table['TP'] = [int(i in db) for i in zip(info_table['feature'], info_table['metabolite'])]

    # Correlation
    cor = [pearson(ys.loc[conditions, m], xs.loc[conditions, k]) for m, k in info_table[['metabolite', 'feature']].values]
    info_table['pearson'], info_table['pearson_pvalue'] = zip(*cor)[:2]
    info_table['pearson_abs'] = info_table['pearson'].abs()
    info_table['pearson_abs_inv'] = 1 - info_table['pearson'].abs()
    info_table['pearson_pvalue_log10'] = np.log10(info_table['pearson_pvalue'])

    # Other metrics
    info_table['euclidean'] = [metric(euclidean_distances, ys.ix[conditions, m], xs.ix[conditions, k])[0][0] for k, m in info_table[['feature', 'metabolite']].values]
    info_table['manhattan'] = [metric(manhattan_distances, ys.ix[conditions, m], xs.ix[conditions, k])[0][0] for k, m in info_table[['feature', 'metabolite']].values]
    info_table['linear_kernel'] = [metric(linear_kernel, ys.ix[conditions, m], xs.ix[conditions, k])[0][0] for k, m in info_table[['feature', 'metabolite']].values]
    info_table['linear_kernel_abs'] = info_table['linear_kernel'].abs()

    info_table = info_table.dropna()
    print '[INFO] TP (%s): %d / %d' % (descprition, info_table['TP'].sum(), info_table.shape[0])

    # ---- Kinase/Enzyme enrichment
    int_enrichment, thresholds = [], roc_curve(info_table['TP'], info_table['score'])[2]

    M = set(zip(info_table['feature'], info_table['metabolite']))
    n = M.intersection(db)

    for threshold in thresholds:
        N = info_table.loc[info_table['score'] > threshold]
        N = set(zip(N['feature'], N['metabolite'])).intersection(M)

        x = N.intersection(n)

        fraction = float(len(x)) / float(len(N)) if float(len(N)) != 0.0 else 0
        p_value = hypergeom.sf(len(x), len(M), len(n), len(N))

        int_enrichment.append((threshold, fraction, p_value, len(M), len(n), len(N), len(x)))

    int_enrichment = DataFrame(int_enrichment, columns=['thres', 'fraction', 'pvalue', 'M', 'n', 'N', 'x']).dropna()
    print '[INFO] Kinase/Enzyme enrichment ready'

    # ---- Plot Kinase/Enzyme enrichment
    # ROC plot analysis
    ax = plot[pos][0]
    for roc_metric in ['score', 'pearson_abs']:
        curve_fpr, curve_tpr, thresholds = roc_curve(info_table['TP'], info_table[roc_metric])
        curve_auc = auc(curve_fpr, curve_tpr)

        ax.plot(curve_fpr, curve_tpr, label='%s (area = %0.2f)' % (roc_metric, curve_auc))

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_ylabel(descprition)
    sns.despine(trim=True, ax=ax)
    ax.legend(loc='lower right')

    # Hypergeometric specific thresold analysis
    ax = plot[pos][1]

    plot_df = int_enrichment[['thres', 'fraction']].copy()
    plot_df = plot_df[plot_df['fraction'] != 0]
    plot_df['fraction'] *= 100

    ax.plot(plot_df['thres'], plot_df['fraction'], c='gray')
    ax.set_xlim(plot_df['thres'].min(), plot_df['thres'].max())
    ax.set_ylim(plot_df['fraction'].min(), plot_df['fraction'].max())
    ax.set_ylabel('Fraction (%)')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    sns.despine(ax=ax)

    pos += 1

plt.savefig('%s/reports/kinase_enzyme_enrichment_metabolomics.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plotting done: %s/reports/kinase_enzyme_enrichment_metabolomics.pdf' % wd
