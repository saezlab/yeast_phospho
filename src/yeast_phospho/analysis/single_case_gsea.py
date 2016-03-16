import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas import DataFrame, read_csv, Series
from pymist.enrichment.gsea import gsea
from yeast_phospho.utilities import get_kinases_targets, get_proteins_name

# -- Imports
# Import kinase targets
k_targets = get_kinases_targets()
k_targets = {t: set(k_targets.ix[map(bool, k_targets[t]), t].index) for t in k_targets}

# Import uniprot mapping
acc_name = get_proteins_name()
acc = read_csv('%s/files/yeast_uniprot.txt' % wd, sep='\t', index_col=2)['oln'].to_dict()

# Import phospho FC
phospho = read_csv('%s/tables/pproteomics_dynamic_combination.csv' % wd, index_col=0)
phospho = phospho[[i.split('_')[0] in acc for i in phospho.index]]
phospho.index = ['%s_%s' % (acc[i.split('_')[0]], i.split('_')[1]) for i in phospho.index]

# -- Estimate kinase activities of combination dynamic data
data, feature = phospho['NaCl_2700'].dropna().to_dict(), 'YNL161W'
signature = k_targets[feature]

print gsea(data, signature, 1000, plot_name='%s/reports/single_case_gsea.pdf' % wd, plot_title=acc_name[feature], y1_label='Enrichment score\n(GSEA)', y2_label='Phosphorylation\n(log FC)')
