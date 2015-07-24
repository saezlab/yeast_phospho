import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymist.reader.sbml_reader import read_sbml_model
from sklearn.cross_validation import LeaveOneOut, KFold, Bootstrap, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.decomposition.pca import PCA
from sklearn.manifold.t_sne import TSNE
from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics.regression import mean_squared_error
from sklearn.svm import SVR, LinearSVC, SVC, LinearSVR
from statsmodels.stats.multitest import multipletests
from yeast_phospho import wd
from yeast_phospho.utils import pearson, spearman
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, LassoCV, Ridge, RidgeClassifierCV
from pandas import DataFrame, Series, read_csv, Index, melt

sns.set_style('ticks')

growth = read_csv(wd + 'files/strain_relative_growth_rate.txt', sep='\t', index_col=0)['relative_growth']

s_info = read_csv(wd + 'files/metabolomics/strain_info.tab', sep='\t', index_col=0)

# ---- Import metabolic model
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

# ---- Calculate steady-state reaction activity
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)
metabolomics.index = Index(metabolomics.index, dtype=np.str)

metabolomics_growth = read_csv('%s/tables/metabolomics_steady_state_growth_rate.tab' % wd, sep='\t', index_col=0)
metabolomics_growth.index = Index([str(i) for i in metabolomics_growth.index], dtype=str)

for xs, f in [(metabolomics.copy(), '%s/tables/reaction_activity_steady_state.tab' % wd), (metabolomics_growth.copy(), '%s/tables/reaction_activity_steady_state_with_growth.tab' % wd)]:
    # Import metabolites annotation
    model_met_map = read_csv(wd + 'files/metabolomics/metabolite_mz_map_dobson.txt', sep='\t', index_col='id')
    model_met_map['mz'] = [str(i) for i in model_met_map['mz']]
    model_met_map = model_met_map.drop_duplicates('mz')['mz'].to_dict()
    model_met_map = {k: model_met_map[k] for k in model_met_map if model_met_map[k] in xs.index}

    # Remove metabolites not measured
    r_metabolites = s_matrix[[i in model_met_map for i in s_matrix.index]]
    r_metabolites.index = [model_met_map[i] for i in r_metabolites.index]

    # Remove reactions with only one metabolite
    r_metabolites = r_metabolites.loc[:, r_metabolites.sum() > 1]

    # Remove un-used metabolites and reactions
    r_metabolites = r_metabolites.loc[:, r_metabolites.sum() != 0]
    r_metabolites = r_metabolites[r_metabolites.sum(1) != 0]

    metabolites, strains, reactions = list(r_metabolites.index), list(xs.columns), list(r_metabolites.columns)
    r_activity = DataFrame({s: dict(zip(*(reactions, RidgeCV().fit(r_metabolites.ix[metabolites, reactions], xs.ix[metabolites, s]).coef_))) for s in strains})

    r_activity.to_csv(f, sep='\t')
    print '[INFO] [REACTION ACTIVITY] Exported to: %s' % f


# ---- Calculate dynamic reaction activity
# Import metabolomics
metabolomics_dyn = read_csv(wd + 'tables/dynamic_metabolomics.tab', sep='\t', index_col=0)
metabolomics_dyn.index = Index([str(i) for i in metabolomics_dyn.index], dtype=str)

# Import metabolites map
model_met_map_dyn = read_csv(wd + 'files/metabolomics/metabolite_mz_map_dobson.txt', sep='\t', index_col='id')
model_met_map_dyn['mz'] = ['%.2f' % i for i in model_met_map_dyn['mz']]
model_met_map_dyn = model_met_map_dyn.drop_duplicates('mz')['mz'].to_dict()
model_met_map_dyn = {k: model_met_map_dyn[k] for k in model_met_map_dyn if model_met_map_dyn[k] in metabolomics_dyn.index}

# Remove metabolites not measured
r_metabolites_dyn = s_matrix[[i in model_met_map_dyn for i in s_matrix.index]]
r_metabolites_dyn.index = [model_met_map_dyn[i] for i in r_metabolites_dyn.index]

# Remove reactions with only one metabolite
r_metabolites_dyn = r_metabolites_dyn.loc[:, r_metabolites_dyn.sum() > 1]

# Remove un-used metabolites and reactions
r_metabolites_dyn = r_metabolites_dyn.loc[:, r_metabolites_dyn.sum() != 0]
r_metabolites_dyn = r_metabolites_dyn[r_metabolites_dyn.sum(1) != 0]

metabolites_dyn, conditions_dyn, reactions_dyn = list(r_metabolites_dyn.index), list(metabolomics_dyn.columns), list(r_metabolites_dyn.columns)

# Calculate reaction activity
r_activity_dyn = DataFrame({c: dict(zip(*(reactions_dyn, RidgeCV().fit(r_metabolites_dyn.ix[metabolites_dyn, reactions_dyn], metabolomics_dyn.ix[metabolites_dyn, c]).coef_))) for c in conditions_dyn})

# Export reaction activities
r_activity_file = '%s/tables/reaction_activity_dynamic.tab' % wd
r_activity_dyn.to_csv(r_activity_file, sep='\t')
print '[INFO] [REACTION ACTIVITY] Exported to: %s' % r_activity_file