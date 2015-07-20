import numpy as np
import seaborn as sns
from yeast_phospho import wd
from yeast_phospho.utils import spearman
from pandas import read_csv

sns.set_style('ticks')

# Import TF and Kinase activity
tf_activity = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0)
k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)

strains = set(tf_activity.columns).intersection(k_activity.columns)

# Import gene names
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)['gene'].to_dict()

# Import TF regulatory sites


def get_site_function(f):
    if 'Inhibits The Protein Function' in f:
        return 'Inhibition'
    elif 'Activates The Protein Function' in f:
        return 'Activation'
    else:
        return np.NaN

tf_reg_sites = read_csv(wd + 'files/PhosphoGrid.txt', sep='\t')[['ORF_NAME', 'PHOSPHO_SITE', 'KINASES_ORFS', 'PHOSPHATASES_ORFS', 'SITE_FUNCTIONS']]
tf_reg_sites = tf_reg_sites.loc[np.bitwise_or(tf_reg_sites['KINASES_ORFS'] != '-', tf_reg_sites['PHOSPHATASES_ORFS'] != '-')]
tf_reg_sites['FUNCTION'] = [get_site_function(f) for f in tf_reg_sites['SITE_FUNCTIONS']]
tf_reg_sites = tf_reg_sites.dropna(subset=['FUNCTION'])
tf_reg_sites = tf_reg_sites[[i in tf_activity.index for i in tf_reg_sites['ORF_NAME']]]
tf_reg_sites['SOURCE'] = tf_reg_sites['KINASES_ORFS'] + '|' + tf_reg_sites['PHOSPHATASES_ORFS']
tf_reg_sites = [(k, r['ORF_NAME'], r['FUNCTION']) for i, r in tf_reg_sites.iterrows() for k in r['SOURCE'].split('|') if k != '-' and r['ORF_NAME'] != k]
tf_reg_sites = {(k, s, f) for k, s, f in tf_reg_sites if k in k_activity.index and s in tf_activity.index}
print '[INFO] Regulatory TF sites extracted'

# ---- Correlation Kinases vs TF
k_tf_cor = [(k, tf, spearman(k_activity.ix[k, strains], tf_activity.ix[tf, strains]), f) for k, tf, f in tf_reg_sites]
