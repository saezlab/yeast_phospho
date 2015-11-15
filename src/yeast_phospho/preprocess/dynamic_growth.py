import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from sklearn.decomposition import PCA
from yeast_phospho.utilities import pearson
from pandas import DataFrame, Series, read_csv, melt
from scipy.interpolate.interpolate import interp1d

p_timepoints = [-10, 5, 9, 15, 25, 44, 79]


growth = read_csv('%s/tables/dynamic_growth.txt' % wd, sep='\t')
growth['time_perturbation_min'] = growth['time_perturbation'] * 60


def interpolate_growth(df, timepoints):
    growth_tp = []
    for sample in set(df['sample']):
        x = df.loc[df['sample'] == sample, 'time_perturbation_min']
        y = df.loc[df['sample'] == sample, 'OD600']

        regression = interp1d(x, y)

        growth_tp.append(regression(timepoints))

    growth_tp = DataFrame(growth_tp).T.median(1)
    growth_tp.index = timepoints

    return growth_tp

growth_tp = DataFrame({cond: interpolate_growth(growth[growth['condition'] == cond], p_timepoints) for cond in set(growth['condition'])})
growth_tp['timepoint'] = growth_tp.index
growth_tp = melt(growth_tp, id_vars='timepoint', value_name='relative_growth')
growth_tp = growth_tp[growth_tp['timepoint'] != -10]
growth_tp['condition'] = [v + '_' + str(t) + 'min' for v, t in growth_tp[['variable', 'timepoint']].values]
growth_tp[['condition', 'relative_growth']].to_csv('%s/files/dynamic_growth.txt' % wd, sep='\t', index=False)


metabolomics = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0)
pca = PCA(10).fit(metabolomics.T)
pcs = pca.transform(metabolomics.T)

x, y = growth_tp['value'].values, pcs[:, 1]

sns.set(style='ticks')
sns.jointplot(x, y, kind='reg', marginal_kws={'hist': False}, xlim=(0.8, 1.5), ylim=(-4, 12))
plt.axhline(y=0, ls='--', c='.5', lw=.3)
plt.axvline(x=0, ls='--', c='.5', lw=.3)
plt.xlabel('OD600')
plt.ylabel('Metabolomics PC1 (%.1f %%)' % (pca.explained_variance_ratio_[0] * 100))
plt.savefig('%s/reports/dynamic_metabolomics_growth_pca.pdf' % wd, bbox_inches='tight')
plt.close('all')
