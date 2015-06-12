import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, Series, read_csv
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, auc, jaccard_similarity_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC

wd = '/Users/emanuel/Projects/projects/yeast_phospho/'

sns.set_style('white')

# Import phospho log2 FC
phospho_df = read_csv(wd + 'tables/steady_state_phosphoproteomics.tab', sep='\t', index_col='site')

# Import regulatory sites
reg_sites = read_csv(wd + 'files/phosphosites.txt', sep='\t')[['ORF_NAME', 'PHOSPHO_SITE', 'SITE_FUNCTIONS', 'KINASES_ORFS', 'PHOSPHATASES_ORFS', 'SITE_CONDITIONS']]
reg_sites = reg_sites[reg_sites['SITE_FUNCTIONS'] != '-']
reg_sites['SITE'] = reg_sites['ORF_NAME'] + '_' + reg_sites['PHOSPHO_SITE']
reg_sites = reg_sites.set_index('SITE')

reg_sites = reg_sites[['Activates The Protein Function' in f or 'Inhibits The Protein Function' in f for f in reg_sites['SITE_FUNCTIONS']]]
# reg_sites = reg_sites[['Cell Cycle' not in i for i in reg_sites['SITE_CONDITIONS']]]

reg_sites_protein = {s.split('_')[0] for s in reg_sites.index}

# ---- AUC
x = phospho_df.var(1).dropna()
x = x[[i.split('_')[0] in reg_sites_protein for i in x.index]]

y = [int(s in reg_sites.index) for s in x.index]

curve_fpr, curve_tpr, _ = roc_curve(y, x)
curve_auc = auc(curve_fpr, curve_tpr)

plt.plot(curve_fpr, curve_tpr, label='area = %0.2f' % curve_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

plt.savefig(wd + 'reports/predict_regulatory_sites.pdf', bbox_inches='tight')
plt.close('all')

# ---- Classification
x = phospho_df.replace(np.NaN, 0)
x[x < 0] = -1
x[x > 0] = 1

x = x[[i.split('_')[0] in reg_sites_protein for i in x.index]]

y = np.array([int(s in reg_sites.index) for s in x.index])

[jaccard_similarity_score(LinearSVC().fit(x.values[train], y[train]).predict(x.values[test]), y[test]) for train, test in KFold(len(y))]