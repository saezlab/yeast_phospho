import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from yeast_phospho import wd
from yeast_phospho.utils import metric, pearson
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, linear_kernel
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from pandas import DataFrame, read_csv, melt, pivot_table
from scipy.stats.distributions import hypergeom
from pymist.reader.sbml_reader import read_sbml_model


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

# Get reactions metabolites
r_metabolites = {r: set(s_matrix.ix[s_matrix[r] != 0, r].index) for r in s_matrix.columns}

# Gene metabolite association
met_reactions = {kegg_id: set(model.s[met_id].keys()) for met_id, kegg_id in model_met_map.items() if met_id in model.s}
gene_reactions = dict(model.get_reactions_by_genes(model.get_genes()).items())

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

        lb = db['combined_score'].max() * .4
        ub = db['combined_score'].max() * 1.

        db = db[db['combined_score'] > lb]
        db = db[db['combined_score'] < ub]

        db = {(source.split('.')[1], target.split('.')[1]) for source, target in db[['protein1', 'protein2']].values}

        print 'lb: %.2f; ub: %.2f' % (lb, ub)

    db = {(s, t) for s, t in db if s != t}
    print '[INFO] %s: %d' % (bkg_type, len(db))

    db = {(s, x) for s, t in db for x in [s, t] if x in gene_reactions}
    print '[INFO] %s, only enzyme targets: %d' % (bkg_type, len(db))

    db = {(s, r) for s, t in db for r in gene_reactions[t] if r in s_matrix.columns}
    print '[INFO] %s, only enzymatic reactions: %d' % (bkg_type, len(db))

    db = {(s, m) for s, r in db if r in s_matrix.columns for m in r_metabolites[r]}
    print '[INFO] %s, only enzymatic reactions metabolites: %d' % (bkg_type, len(db))

    db = {(s, model_met_map[m]) for s, m in db if m in model_met_map}
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
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)['gene'].to_dict()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}


# ---- Import data-sets
# Steady-state
metabolomics_std = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0).ix[cmap.columns].dropna()
metabolomics_std = metabolomics_std[metabolomics_std.std(1) > .4]

k_activity_std = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0).ix[cmap.index]
k_activity_std = k_activity_std[(k_activity_std.count(1) / k_activity_std.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_std = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0).ix[cmap.index].dropna()
tf_activity_std = tf_activity_std[tf_activity_std.std(1) > .3]

# Steady-state with growth
metabolomics_std_g = read_csv('%s/tables/metabolomics_steady_state_growth_rate.tab' % wd, sep='\t', index_col=0).ix[cmap.columns].dropna()
metabolomics_std_g = metabolomics_std_g[metabolomics_std_g.std(1) > .4]

k_activity_std_g = read_csv('%s/tables/kinase_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0).ix[cmap.index]
k_activity_std_g = k_activity_std_g[(k_activity_std_g.count(1) / k_activity_std_g.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_std_g = read_csv('%s/tables/tf_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0).ix[cmap.index].dropna()
tf_activity_std_g = tf_activity_std_g[tf_activity_std_g.std(1) > .3]

# Dynamic
metabolomics_dyn = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0).ix[cmap.columns].dropna()
metabolomics_dyn = metabolomics_dyn[metabolomics_dyn.std(1) > .4]

k_activity_dyn = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0).ix[cmap.index]
k_activity_dyn = k_activity_dyn[(k_activity_dyn.count(1) / k_activity_dyn.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_dyn = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0).ix[cmap.index].dropna()
tf_activity_dyn = tf_activity_dyn[tf_activity_dyn.std(1) > .3]


# ---- Overlap
strains = list(set(metabolomics_std.columns).intersection(k_activity_std.columns).intersection(tf_activity_std.columns))
conditions = list(set(metabolomics_dyn.columns).intersection(k_activity_dyn.columns).intersection(tf_activity_dyn.columns))
metabolites = list(set(metabolomics_std.index).intersection(metabolomics_dyn.index))
kinases = list(set(k_activity_std.index).intersection(k_activity_dyn.index))
tfs = list(set(tf_activity_std.index).intersection(tf_activity_dyn.index))

metabolomics_std, k_activity_std, tf_activity_std = metabolomics_std.ix[metabolites, strains], k_activity_std.ix[kinases, strains], tf_activity_std.ix[tfs, strains]
metabolomics_std_g, k_activity_std_g, tf_activity_std_g = metabolomics_std_g.ix[metabolites, strains], k_activity_std_g.ix[kinases, strains], tf_activity_std_g.ix[tfs, strains]
metabolomics_dyn, k_activity_dyn, tf_activity_dyn = metabolomics_dyn.ix[metabolites, conditions], k_activity_dyn.ix[kinases, conditions], tf_activity_dyn.ix[tfs, conditions]
print '[INFO] Data-sets import and overlap calculated'


# ---- Comparisons
datasets_files = [
    (k_activity_std, metabolomics_std, 'Kinase steadystate'),
    (tf_activity_std, metabolomics_std, 'TF steadystate'),

    (k_activity_std_g, metabolomics_std_g, 'Kinase steadystate (with growth)'),
    (tf_activity_std_g, metabolomics_std_g, 'TF steadystate (with growth)'),

    (k_activity_dyn, metabolomics_dyn, 'Kinase dynamic'),
    (tf_activity_dyn, metabolomics_dyn, 'TF dynamic'),
]


# ---- Perform kinase/enzyme enrichment
sns.set(style='ticks', palette='pastel', color_codes=True, context='paper')
(f, plot), pos = plt.subplots(len(datasets_files), 2, figsize=(7, 4. * len(datasets_files))), 0
for k_activity, metabolomics, growth in datasets_files:

    # Overlapping kinases/phosphatases knockout
    strains = list(set(k_activity.columns).intersection(set(metabolomics.columns)))
    k_activity, metabolomics = k_activity[strains], metabolomics[strains]

    # ---- Correlate metabolic fold-changes with kinase activities
    lm = Lasso(alpha=0.01, max_iter=2000).fit(k_activity.T, metabolomics.T)

    info_table = DataFrame(lm.coef_, index=metabolomics.index, columns=k_activity.index)
    info_table['metabolite'] = info_table.index
    info_table = melt(info_table, id_vars='metabolite', value_name='coef', var_name='kinase')
    info_table = info_table[info_table['coef'] != 0.0]

    info_table['score'] = info_table['coef'].abs().max() - info_table['coef'].abs()

    info_table['kinase_count'] = [k_activity.ix[k].count() for k in info_table['kinase']]

    info_table['metabolite_name'] = [m_map[m] if m in m_map else str(m) for m in info_table['metabolite']]
    info_table['kinase_name'] = [acc_name[k] if k in acc_name else str(k) for k in info_table['kinase']]

    # Kinase/Enzyme interactions via metabolite correlations
    info_table['TP'] = [int(i in db) for i in zip(info_table['kinase'], info_table['metabolite'])]

    # Correlation
    cor = [pearson(metabolomics.loc[m, strains], k_activity.loc[k, strains]) for m, k in info_table[['metabolite', 'kinase']].values]
    info_table['pearson'], info_table['pearson_pvalue'] = zip(*cor)[:2]
    info_table['pearson_abs'] = info_table['pearson'].abs()
    info_table['pearson_abs_inv'] = 1 - info_table['pearson'].abs()
    info_table['pearson_pvalue_log10'] = np.log10(info_table['pearson_pvalue'])

    # Other metrics
    info_table['euclidean'] = [metric(euclidean_distances, metabolomics.ix[m, strains], k_activity.ix[k, strains])[0][0] for k, m in zip(info_table['kinase'], info_table['metabolite'])]
    info_table['manhattan'] = [metric(manhattan_distances, metabolomics.ix[m, strains], k_activity.ix[k, strains])[0][0] for k, m in zip(info_table['kinase'], info_table['metabolite'])]
    info_table['linear_kernel'] = [metric(linear_kernel, metabolomics.ix[m, strains], k_activity.ix[k, strains])[0][0] for k, m in zip(info_table['kinase'], info_table['metabolite'])]
    info_table['linear_kernel_abs'] = info_table['linear_kernel'].abs()

    info_table = info_table.dropna()
    print '[INFO] TP (%s): %d / %d' % (growth, info_table['TP'].sum(), info_table.shape[0])

    # ---- Kinase/Enzyme enrichment
    int_enrichment, thresholds = [], roc_curve(info_table['TP'], info_table['score'])[2]

    M = set(zip(info_table['kinase'], info_table['metabolite']))
    n = M.intersection(db)

    for threshold in thresholds:
        N = info_table.loc[info_table['score'] > threshold]
        N = set(zip(N['kinase'], N['metabolite'])).intersection(M)

        x = N.intersection(n)

        fraction = float(len(x)) / float(len(N)) if float(len(N)) != 0.0 else 0
        p_value = hypergeom.sf(len(x), len(M), len(n), len(N))

        int_enrichment.append((threshold, fraction, p_value, len(M), len(n), len(N), len(x)))

        print threshold, fraction, p_value, len(M), len(n), len(N), len(x)

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
    ax.set_ylabel(growth)
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
