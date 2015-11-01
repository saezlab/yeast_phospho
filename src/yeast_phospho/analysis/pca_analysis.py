import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from sklearn.decomposition.factor_analysis import FactorAnalysis
from pandas import DataFrame, Series, read_csv
from yeast_phospho.utilities import get_ko_strains, pearson


# Import growth rates
growth = read_csv('%s/files/strain_relative_growth_rate.txt' % wd, sep='\t', index_col=0)['relative_growth']
ko_strains = list(growth.index)


# Import data-sets
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)[ko_strains]

trans = read_csv('%s/tables/transcriptomics_steady_state.tab' % wd, sep='\t', index_col=0).loc[:, ko_strains].dropna(how='all', axis=1)

phospho_df = read_csv('%s/tables/pproteomics_steady_state.tab' % wd, sep='\t', index_col=0).loc[:, ko_strains].dropna(how='all', axis=1)
phospho_df = phospho_df[(phospho_df.count(1) / phospho_df.shape[1]) > .25].replace(np.NaN, 0.0)

# PCA analysis
for df in [metabolomics.T, phospho_df.T, trans.T]:
    # pca = PCA(n_components=10).fit(df)
    pca = FactorAnalysis(n_components=10).fit(df)
    pcs = DataFrame(pca.transform(df), index=df.index, columns=['PC%d' % (i+1) for i in range(10)])
    # var = pca.explained_variance_ratio_

    print [(pc, pearson(pcs.ix[ko_strains, pc], growth[ko_strains])) for pc in ['PC%d' % (i+1) for i in range(10)]]
