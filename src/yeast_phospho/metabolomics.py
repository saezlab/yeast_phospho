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

metabol_df = read_csv('%s/data/steady_state_metabolomics.tab' % wd, sep='\t').dropna()
metabol_df.index = Index(metabol_df['m/z'], dtype=np.str)
metabol_df = metabol_df.drop('m/z', 1).dropna()

fc_thres, n_fc_thres = 0.8, 0
metabol_df = metabol_df[(metabol_df.abs() > fc_thres).sum(1) > n_fc_thres]

# ---- Import metabolic model
model_met_map = read_csv(wd + 'files/metabolomics/metabolite_mz_map_dobson.txt', sep='\t', index_col='id')
model_met_map['mz'] = [str(i) for i in model_met_map['mz']]
model_met_map = model_met_map.drop_duplicates('mz')['mz'].to_dict()
model_met_map = {k: model_met_map[k] for k in model_met_map if model_met_map[k] in metabol_df.index}

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

# Remove metabolites not measured
s_matrix = s_matrix[[i in model_met_map for i in s_matrix.index]]

# Map metabolite to m/z
s_matrix.index = [model_met_map[i] for i in s_matrix.index]

# Remove reactions with only one metabolite
s_matrix = s_matrix.loc[:, s_matrix.sum() > 1]

# Remove un-used metabolites and reactions
s_matrix = s_matrix.loc[:, s_matrix.sum() != 0]
s_matrix = s_matrix[s_matrix.sum(1) != 0]


# ---- Calculate reaction activity
metabolites, strains, reactions = list(s_matrix.index), list(metabol_df.columns), list(s_matrix.columns)

r_activity = DataFrame({s: dict(zip(*(reactions, LinearRegression().fit(s_matrix.ix[metabolites, reactions], metabol_df.ix[metabolites, s]).coef_))) for s in strains})
# r_activity = DataFrame(LinearRegression().fit(s_matrix.ix[metabolites, reactions], metabol_df.ix[metabolites, strains]).coef_, index=strains, columns=reactions).T

y = Series([int(i in ['high', 'average']) for i in s_info['impact']], index=s_info.index)
x = r_activity[y.index].T

roc_pred = [roc_auc_score(y.ix[test].values, RidgeClassifierCV().fit(x.ix[train], y.ix[train]).decision_function(x.ix[test])) for train, test in StratifiedShuffleSplit(y.values, n_iter=1000)]
print np.median(roc_pred)

y_pred = r_activity[y.index].std()
print roc_auc_score(y, y_pred)


# ---- Predict reaction activity
k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)
strains = list(set(k_activity.columns).intersection(metabol_df.columns))

x, y = k_activity.loc[:, strains].replace(np.NaN, 0.0).T, r_activity.loc[reactions, strains].T

r_predicted = DataFrame({strains[test]: {r: RidgeCV().fit(x.ix[train], y.ix[train, r]).predict(x.ix[test])[0] for r in reactions} for train, test in LeaveOneOut(len(x))})

# Plot predicted prediction scores
m_score = [(r, spearman(r_activity.ix[r, strains], r_predicted.ix[r, strains])) for r in reactions]
m_score = DataFrame([(m, c, p, n) for m, (c, p, n) in m_score], columns=['reaction', 'correlation', 'pvalue', 'n_meas'])
m_score['adjpvalue'] = multipletests(m_score['pvalue'], method='fdr_bh')[1]
m_score = m_score.sort('correlation', ascending=False)
print 'Mean correlation metabolites: ', m_score['correlation'].mean()

s_score = [(s, spearman(r_activity.ix[reactions, s], r_predicted.ix[reactions, s])) for s in strains]
s_score = DataFrame([(m, c, p, n) for m, (c, p, n) in s_score], columns=['strain', 'correlation', 'pvalue', 'n_meas'])
s_score['adjpvalue'] = multipletests(s_score['pvalue'], method='fdr_bh')[1]
s_score = s_score.sort('correlation', ascending=False)
print 'Mean correlation samples: ', s_score['correlation'].mean()