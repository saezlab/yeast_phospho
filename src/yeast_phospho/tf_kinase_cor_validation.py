import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from yeast_phospho.utils import spearman
from pandas import DataFrame, read_csv, pivot_table
from sklearn.metrics import roc_curve, auc

sns.set_style('ticks')

# Import TF and Kinase activity
tf_activity = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0)
k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)

strains = set(tf_activity.columns).intersection(k_activity.columns)

# Import gene names
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)['gene'].to_dict()

# Import TF regulatory sites
tf_reg_sites = read_csv(wd + 'files/PhosphoGrid.txt', sep='\t')[['ORF_NAME', 'PHOSPHO_SITE', 'KINASES_ORFS', 'PHOSPHATASES_ORFS', 'SITE_FUNCTIONS']]
tf_reg_sites = tf_reg_sites.loc[np.bitwise_or(tf_reg_sites['KINASES_ORFS'] != '-', tf_reg_sites['PHOSPHATASES_ORFS'] != '-')]
tf_reg_sites = tf_reg_sites[['Inhibits The Protein Function' in f or 'Activates The Protein Function' in f for f in tf_reg_sites['SITE_FUNCTIONS']]]
tf_reg_sites = tf_reg_sites[[i in tf_activity.index for i in tf_reg_sites['ORF_NAME']]]
tf_reg_sites['SOURCE'] = tf_reg_sites['KINASES_ORFS'] + '|' + tf_reg_sites['PHOSPHATASES_ORFS']
tf_reg_sites = [(k, r['ORF_NAME']) for i, r in tf_reg_sites.iterrows() for k in r['SOURCE'].split('|') if k != '-' and r['ORF_NAME'] != k]
tf_reg_sites = {(k, s) for k, s in tf_reg_sites if k in k_activity.index and s in tf_activity.index}
print '[INFO] Regulatory TF sites extracted'

# ---- Correlation Kinases vs TF
k_tf_cor = [(k, tf, spearman(k_activity.ix[k, strains], tf_activity.ix[tf, strains])) for k, tf in tf_reg_sites]