import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from matplotlib_venn import venn3, venn3_circles
from pandas import DataFrame, Series, read_csv, pivot_table

# -- Steady-state
# transcriptomics
name2id = read_csv('%s/files/orf_name_dataframe.tab' % wd, sep='\t', index_col=1).to_dict()['orf']
id2name = read_csv('%s/files/orf_name_dataframe.tab' % wd, sep='\t', index_col=0).to_dict()['name']

transcriptomics = read_csv('%s/data/Kemmeren_2014_zscores_parsed_filtered.tab' % wd, sep='\t', header=False)
transcriptomics['tf'] = [name2id[i] if i in name2id else id2name[i] for i in transcriptomics['tf']]
transcriptomics = pivot_table(transcriptomics, values='value', index='target', columns='tf').dropna(how='all', axis=1)

# phosphoprotoemics
phospho_df = read_csv('%s/data/steady_state_phosphoproteomics.tab' % wd, sep='\t')
phospho_df = phospho_df.pivot_table(values='logFC', index=['peptide', 'target'], columns='regulator', aggfunc=np.median).dropna(how='all', axis=1)

# metabolomics
metabol_df = read_csv(wd + 'data/steady_state_metabolomics.tab', sep='\t').dropna()

# -- Plot
trans, phosp, metab = set(transcriptomics), set(phospho_df), set(metabol_df)
venn3([trans, phosp, metab], set_labels=('Transcriptomics', 'Phosphoproteomics', 'Metabolomics'))
plt.savefig('%s/reports/steady_state_overlap_conditions_venn.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Plot done'
