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

metabolomics = read_csv('%s/data/steady_state_metabolomics.tab' % wd, sep='\t').dropna()
metabolomics.index = Index(metabolomics['m/z'], dtype=np.str)
metabolomics = metabolomics.drop('m/z', 1).dropna()

fc_thres, n_fc_thres = 0.8, 0
metabolomics = metabolomics[(metabolomics.abs() > fc_thres).sum(1) > n_fc_thres]

# ---- Import metabolic model
model_met_map = read_csv(wd + 'files/metabolomics/metabolite_mz_map_dobson.txt', sep='\t', index_col='id')
model_met_map['mz'] = [str(i) for i in model_met_map['mz']]
model_met_map = model_met_map.drop_duplicates('mz')['mz'].to_dict()
model_met_map = {k: model_met_map[k] for k in model_met_map if model_met_map[k] in metabolomics.index}

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


# ---- Calculate reaction activity
# Remove metabolites not measured
r_metabolites = s_matrix[[i in model_met_map for i in s_matrix.index]]
r_metabolites.index = [model_met_map[i] for i in r_metabolites.index]

# Remove reactions with only one metabolite
r_metabolites = r_metabolites.loc[:, r_metabolites.sum() > 1]

# Remove un-used metabolites and reactions
r_metabolites = r_metabolites.loc[:, r_metabolites.sum() != 0]
r_metabolites = r_metabolites[r_metabolites.sum(1) != 0]

metabolites, strains, reactions = list(r_metabolites.index), list(metabolomics.columns), list(r_metabolites.columns)
r_activity = DataFrame({s: dict(zip(*(reactions, LinearRegression().fit(r_metabolites.ix[metabolites, reactions], metabolomics.ix[metabolites, s]).coef_))) for s in strains})

y = Series([int(i in ['high', 'average']) for i in s_info['impact']], index=s_info.index)
x = r_activity[y.index].T

roc_pred = [roc_auc_score(y.ix[test].values, RidgeClassifierCV().fit(x.ix[train], y.ix[train]).decision_function(x.ix[test])) for train, test in StratifiedShuffleSplit(y.values, n_iter=1000)]
print np.median(roc_pred)

y_pred = r_activity[y.index].std()
print roc_auc_score(y, y_pred)


# ---- Predict reaction activity
k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)
strains = list(set(k_activity.columns).intersection(metabolomics.columns))

x, y = k_activity.loc[:, strains].replace(np.NaN, 0.0).T, r_activity.loc[reactions, strains].T

r_predicted = DataFrame({strains[test]: {r: Ridge().fit(x.ix[train], y.ix[train, r]).predict(x.ix[test])[0] for r in reactions} for train, test in LeaveOneOut(len(x))})

# Plot predicted prediction scores
r_score = [(r, pearson(r_activity.ix[r, strains], r_predicted.ix[r, strains])) for r in reactions]
r_score = DataFrame([(m, c, p, n) for m, (c, p, n) in r_score], columns=['reaction', 'correlation', 'pvalue', 'n_meas'])
r_score['adjpvalue'] = multipletests(r_score['pvalue'], method='fdr_bh')[1]
r_score = r_score.sort('correlation', ascending=False)
print 'Mean correlation metabolites: ', r_score['correlation'].mean()

s_score = [(s, pearson(r_activity.ix[reactions, s], r_predicted.ix[reactions, s])) for s in strains]
s_score = DataFrame([(m, c, p, n) for m, (c, p, n) in s_score], columns=['strain', 'correlation', 'pvalue', 'n_meas'])
s_score['adjpvalue'] = multipletests(s_score['pvalue'], method='fdr_bh')[1]
s_score = s_score.sort('correlation', ascending=False)
print 'Mean correlation samples: ', s_score['correlation'].mean()


# ---- Dynamic: predict metabolites FC with kinases
# Import kinase activity
k_activity_dyn = read_csv(wd + 'tables/kinase_activity_dynamic.tab', sep='\t', index_col=0)

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

kinases_ov = list(set(k_activity.index).intersection(k_activity_dyn.index))
metabol_ov = list(set(['%.2f' % float(i) for i in metabolomics.index]).intersection(r_metabolites_dyn.index))
conditions = list(metabolomics_dyn.columns)
reactions_ov = list(set(r_metabolites.columns).intersection(r_metabolites_dyn.columns))


# Calculate reaction activity
r_activity_dyn = DataFrame({c: dict(zip(*(reactions_ov, LinearRegression().fit(r_metabolites_dyn.ix[metabol_ov, reactions_ov], metabolomics_dyn.ix[metabol_ov, c]).coef_))) for c in conditions})


# Fit linear regression model
x_train, y_train = k_activity.ix[kinases_ov, strains].dropna(how='all').replace(np.NaN, 0.0).T, r_activity.ix[reactions_ov, strains].T
x_test, y_test = k_activity_dyn.ix[kinases_ov, conditions].replace(np.NaN, 0.0).T, r_activity_dyn.ix[reactions_ov, conditions].T

r_predicted_dyn = DataFrame({r: dict(zip(*(conditions, Ridge().fit(x_train, y_train[r]).predict(x_test.ix[conditions])))) for r in reactions_ov}).T


# Plot predicted prediction scores
r_score_dyn = [(r, pearson(r_activity_dyn.ix[r, conditions], r_predicted_dyn.ix[r, conditions])) for r in reactions_ov]
r_score_dyn = DataFrame([(m, c, p, n) for m, (c, p, n) in r_score_dyn], columns=['reaction', 'correlation', 'pvalue', 'n_meas']).dropna()
r_score_dyn = r_score_dyn.set_index('reaction')
r_score_dyn['adjpvalue'] = multipletests(r_score_dyn['pvalue'], method='fdr_bh')[1]
r_score_dyn['signif'] = ['FDR < 5%' if x < 0.05 else ' FDR >= 5%' for x in r_score_dyn['adjpvalue']]
r_score_dyn = r_score_dyn.sort('correlation', ascending=False)
print 'Mean correlation metabolites: ', r_score_dyn['correlation'].mean()

s_score_dyn = [(s, pearson(r_activity_dyn.ix[reactions_ov, s], r_predicted_dyn.ix[reactions_ov, s])) for s in conditions]
s_score_dyn = DataFrame([(m, c, p, n) for m, (c, p, n) in s_score_dyn], columns=['condition', 'correlation', 'pvalue', 'n_meas']).dropna()
s_score_dyn = s_score_dyn.set_index('condition')
s_score_dyn['adjpvalue'] = multipletests(s_score_dyn['pvalue'], method='fdr_bh')[1]
s_score_dyn['signif'] = ['FDR < 5%' if x < 0.05 else ' FDR >= 5%' for x in s_score_dyn['adjpvalue']]
s_score_dyn = s_score_dyn.sort('correlation', ascending=False)
print 'Mean correlation samples: ', s_score_dyn['correlation'].mean()
