import pickle
import numpy as np
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas import DataFrame, Series, read_csv
from pymist.reader.sbml_reader import read_sbml_model
from yeast_phospho.utilities import get_kinases_targets, get_tfs_targets


# -- Import targets
# Import kinase targets
k_targets = get_kinases_targets()
k_targets = {t: set(k_targets.ix[map(bool, k_targets[t]), t].index) for t in k_targets}
k_targets = {k: {t.split('_')[0] for t in k_targets[k]} for k in k_targets}

# Import TF targets
tf_targets = get_tfs_targets()
tf_targets = {t: set(tf_targets.ix[map(bool, tf_targets[t]), t].index) for t in tf_targets}
tf_targets = {tf: {t.split('_')[0] for t in tf_targets[tf]} for tf in tf_targets}


# -- Background population
all_kinases = set(read_csv('%s/tables/kinase_activity_dynamic_gsea.tab' % wd, sep='\t', index_col=0).index)
all_metabolites = {'%.4f' % i for i in read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0).index}
all_tfs = set(read_csv('%s/tables/tf_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0).index)


# -- Import metabolic model ion mapping
annot = read_csv('%s/files/Annotation_Yeast_glucose.csv' % wd, sep=',', index_col=1)
annot['mz'] = ['%.4f' % round(i, 2) for i in annot['mz']]
annot = annot['mz'].to_dict()

# -- Import metabolic model
model = read_sbml_model('/Users/emanuel/Projects/resources/metabolic_models/iMM904.v1.xml')

# Remove extracellular metabolites
s_matrix = model.get_stoichiometric_matrix()
s_matrix = s_matrix[[not i.endswith('_b') for i in s_matrix.index]]

# Remove biomass reactions
s_matrix = s_matrix.drop('R_biomass_SC5_notrace', axis=1)

# Build {metabolite: protein} dict
m_dict = {i: {r for r in s_matrix.loc[i, s_matrix.ix[i] != 0].index if not r.startswith('R_EX_')} for i in s_matrix.index}
m_dict = {m: {g for r in m_dict[m] for g in model.get_reaction_genes(r)} for m in m_dict}
m_dict = {m: {g for x in m_dict if x[2:-2] == m for g in m_dict[x]} for m in annot}
m_dict = {m: m_dict[m] for m in m_dict if 0 < len(m_dict[m])}

# Filter highly connected metabolites
hmet = {
    'pi', 'ppi',
    'h2o', 'h', 'o2', 'co2',
    'adp', 'atp', 'gtp', 'imp', 'amp', 'ctp', 'ump', 'udp', 'utp', 'gmp', 'gdp',
    'coa', 'accoa'
    'nad', 'nadh', 'nadp', 'nadph', 'nadph',
    'so4', 'udpg', 'dudp',
    'hdca'
}
m_dict = {m: m_dict[m] for m in m_dict if m not in hmet}

# Build {protein: metabolite} dict
m_genes = {g for m in m_dict for g in m_dict[m]}
g_dict = {g: {m for m in m_dict if g in m_dict[m]} for g in m_genes}

# Build {protein: ion} dict
i_dict = {g: {annot[m] for m in g_dict[g]} for g in g_dict}


# -- Read protein interactions dbs
dbs = {}
for bkg_type in ['kinases', 'tfs']:
    if bkg_type == 'kinases':
        db = {(k, t) for k in k_targets for t in k_targets[k]}

    elif bkg_type == 'tfs':
        db = {(tf, t) for tf in tf_targets for t in tf_targets[tf]}

    db = {(s, i) for s, t in db if t in i_dict for i in i_dict[t]}
    print '[INFO] %s, only enzymatic reactions: %d' % (bkg_type, len(db))

    dbs[bkg_type] = db

print '[INFO] Kinase-TFs/Enzymes interactions data-bases imported'


# -- Export results
with open('%s/tables/known_associations.pickle' % wd, 'wb') as handle:
    pickle.dump(dbs, handle, protocol=pickle.HIGHEST_PROTOCOL)
