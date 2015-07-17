import numpy as np
from yeast_phospho import wd
from pandas import DataFrame, read_csv, pivot_table


# Import kinases targets dictionary
k_targets = read_csv('%s/files/PhosphoGrid.txt' % wd, sep='\t').loc[:, ['ORF_NAME', 'PHOSPHO_SITE', 'KINASES_ORFS', 'PHOSPHATASES_ORFS', 'SEQUENCE']]
k_targets = k_targets.loc[np.bitwise_or(k_targets['KINASES_ORFS'] != '-', k_targets['PHOSPHATASES_ORFS'] != '-')]
k_targets['SOURCE'] = k_targets['KINASES_ORFS'] + '|' + k_targets['PHOSPHATASES_ORFS']
k_targets = [(k, t + '_' + site) for t, site, source in k_targets[['ORF_NAME', 'PHOSPHO_SITE', 'SOURCE']].values for k in source.split('|') if k != '-' and k != '' and t != k]
k_targets = DataFrame(k_targets, columns=['kinase', 'site'])
k_targets['value'] = 1
k_targets = pivot_table(k_targets, values='value', index='site', columns='kinase', fill_value=0)
print '[INFO] [PHOSPHOGRID] Kinases targets: ', k_targets.shape

# Export kinase targets data-set
k_targets_file = '%s/tables/kinases_targets_phosphogrid.tab' % wd
k_targets.to_csv(k_targets_file, sep='\t')
print '[INFO] [PHOSPHOGRID] Exported to: %s' % k_targets_file
