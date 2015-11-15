import re
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from yeast_phospho import wd
from yeast_phospho.utilities import get_metabolites_model_annot, metric, pearson, get_proteins_name, get_metabolites_name
from pymist.reader.sbml_reader import read_sbml_model
from scipy.stats.distributions import hypergeom
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, linear_kernel
from pandas import DataFrame, read_csv, pivot_table, melt


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


# ---- Import IDs maps
acc_name = get_proteins_name()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}

met_name = get_metabolites_name()
met_name = {k: met_name[k] for k in met_name if len(met_name[k].split('; ')) == 1}


# ---- Import
# Steady-state without growth
metabolomics_ng = read_csv('%s/tables/metabolomics_steady_state_no_growth.tab' % wd, sep='\t', index_col=0)
metabolomics_ng = metabolomics_ng[metabolomics_ng.std(1) > .4]
metabolomics_ng.index = [str(i) for i in metabolomics_ng.index]

k_activity_ng = read_csv('%s/tables/kinase_activity_steady_state_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_ng = k_activity_ng[(k_activity_ng.count(1) / k_activity_ng.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_ng = read_csv('%s/tables/tf_activity_steady_state_no_growth.tab' % wd, sep='\t', index_col=0)
tf_activity_ng = tf_activity_ng[tf_activity_ng.std(1) > .4]


# Dynamic without growth
metabolomics_dyn_ng = read_csv('%s/tables/metabolomics_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
metabolomics_dyn_ng = metabolomics_dyn_ng[metabolomics_dyn_ng.std(1) > .4]
metabolomics_dyn_ng.index = [str(i) for i in metabolomics_dyn_ng.index]

k_activity_dyn_ng = read_csv('%s/tables/kinase_activity_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_ng = k_activity_dyn_ng[(k_activity_dyn_ng.count(1) / k_activity_dyn_ng.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_dyn_ng = read_csv('%s/tables/tf_activity_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
tf_activity_dyn_ng = tf_activity_dyn_ng[tf_activity_dyn_ng.std(1) > .4]

# Linear regression results
with open('%s/tables/linear_regressions.pickle' % wd, 'rb') as handle:
    lm_res = pickle.load(handle)


# ---- Define comparisons
comparisons = [
    (k_activity_ng, metabolomics_ng, 'Kinases', 'Steady-state', 'without'),
    (tf_activity_ng, metabolomics_ng, 'TFs', 'Steady-state', 'without'),

    (k_activity_dyn_ng, metabolomics_dyn_ng, 'Kinases', 'Dynamic', 'without'),
    (tf_activity_dyn_ng, metabolomics_dyn_ng, 'TFs', 'Dynamic', 'without')
]


# ---- Perform kinase/enzyme enrichment
sns.set(style='ticks', palette='pastel', color_codes=True, context='paper')
(f, plot), pos = plt.subplots(len(comparisons), 2, figsize=(7, 4. * len(comparisons))), 0
for xs, ys, ft, dt, gt in comparisons:
    # ---- Define variables
    conditions = list(set(xs).intersection(ys))
    description = ' '.join([ft, dt, gt])

    # ---- Conditions betas
    info_table = DataFrame([i[1][3] for i in lm_res if i[1][0] == ft and i[1][1] == dt and i[1][2] == gt][0])
    info_table['feature'] = info_table.index
    info_table = melt(info_table, id_vars='feature', var_name='metabolite', value_name='coef')
    info_table = info_table[info_table['coef'] != 0.0]

    info_table['score'] = info_table['coef'].abs().max() - info_table['coef'].abs()

    info_table['metabolite_name'] = [met_name[m] if m in met_name else str(m) for m in info_table['metabolite']]
    info_table['feature_name'] = [acc_name[k] if k in acc_name else str(k) for k in info_table['feature']]

    # Kinase/Enzyme interactions via metabolite correlations
    info_table['TP'] = [int(i in db) for i in zip(info_table['feature'], info_table['metabolite'])]

    # Correlation
    cor = [pearson(ys.loc[m, conditions], xs.loc[k, conditions]) for m, k in info_table[['metabolite', 'feature']].values]
    info_table['pearson'], info_table['pearson_pvalue'] = zip(*cor)[:2]
    info_table['pearson_abs'] = info_table['pearson'].abs()
    info_table['pearson_abs_inv'] = 1 - info_table['pearson'].abs()
    info_table['pearson_pvalue_log10'] = np.log10(info_table['pearson_pvalue'])

    # Other metrics
    info_table['euclidean'] = [metric(euclidean_distances, ys.ix[m, conditions], xs.ix[k, conditions])[0][0] for k, m in info_table[['feature', 'metabolite']].values]
    info_table['manhattan'] = [metric(manhattan_distances, ys.ix[m, conditions], xs.ix[k, conditions])[0][0] for k, m in info_table[['feature', 'metabolite']].values]
    info_table['linear_kernel'] = [metric(linear_kernel, ys.ix[m, conditions], xs.ix[k, conditions])[0][0] for k, m in info_table[['feature', 'metabolite']].values]
    info_table['linear_kernel_abs'] = info_table['linear_kernel'].abs()

    info_table = info_table.dropna()
    print '[INFO] TP (%s): %d / %d' % (description, info_table['TP'].sum(), info_table.shape[0])

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
    ax.set_ylabel(description)
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
