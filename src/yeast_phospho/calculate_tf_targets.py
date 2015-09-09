from yeast_phospho import wd
from pandas import read_csv, pivot_table

# Import conversion table
name2id = read_csv('%s/files/orf_name_dataframe.tab' % wd, sep='\t', index_col=1).to_dict()['orf']
id2name = read_csv('%s/files/orf_name_dataframe.tab' % wd, sep='\t', index_col=0).to_dict()['name']

# TF targets
tf_targets = read_csv('%s/files/tf_gene_network_chip_only.tab' % wd, sep='\t')
tf_targets['tf'] = [name2id[i] if i in name2id else id2name[i] for i in tf_targets['tf']]
# tf_targets = tf_targets[tf_targets['tf'] != tf_targets['target']]
tf_targets['interaction'] = 1
tf_targets = pivot_table(tf_targets, values='interaction', index='target', columns='tf', fill_value=0)
print '[INFO] TF targets calculated!'

# Export TF targets data-set
tf_targets_file = '%s/tables/targets_tfs.tab' % wd
tf_targets.to_csv(tf_targets_file, sep='\t')
print '[INFO] [PHOSPHOGRID] Exported to: %s' % tf_targets_file
