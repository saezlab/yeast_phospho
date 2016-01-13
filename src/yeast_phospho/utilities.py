import re
import numpy as np
import itertools as it
import statsmodels.api as sm
import statsmodels.tools as st
from yeast_phospho import wd
from pandas.stats.misc import zscore
from sklearn.linear_model import Ridge
from scipy.stats.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from pymist.enrichment.gsea import gsea
from pandas import Series, DataFrame, read_csv, pivot_table


# -- Kinases and TFs get targets utility functions
def jaccard(a, b):
    return float(len(a.intersection(b))) / float(len(a.union(b)))


def get_tfs_targets(remove_self=False):
    """
    Retrieve transcription-factor targets

    :param remove_self:
    :return:
    """
    # Import conversion table
    name2id = read_csv('%s/files/orf_name_dataframe.tab' % wd, sep='\t', index_col=1).to_dict()['orf']
    id2name = read_csv('%s/files/orf_name_dataframe.tab' % wd, sep='\t', index_col=0).to_dict()['name']

    # TF targets
    tf_targets = read_csv('%s/files/tf_gene_network_chip_only.tab' % wd, sep='\t')
    tf_targets['tf'] = [name2id[i] if i in name2id else id2name[i] for i in tf_targets['tf']]

    if remove_self:
        tf_targets = tf_targets[tf_targets['tf'] != tf_targets['target']]

    tf_targets['interaction'] = 1
    tf_targets = pivot_table(tf_targets, values='interaction', index='target', columns='tf', fill_value=0)

    return tf_targets


def get_kinases_targets(studies_to_filter={'21177495', '19779198'}, remove_self=False):
    """
    Retrieve kinase targets

    :param studies_to_filter:
    :param remove_self:
    :return:
    """
    k_targets = read_csv('%s/files/PhosphoGrid.txt' % wd, sep='\t')

    k_targets = k_targets[[len(set(i.split('|')).intersection(studies_to_filter)) != len(set(i.split('|'))) for i in k_targets['KINASES_EVIDENCE_PUBMED']]]
    k_targets = k_targets[[len(set(i.split('|')).intersection(studies_to_filter)) != len(set(i.split('|'))) for i in k_targets['PHOSPHATASES_EVIDENCE_PUBMED']]]

    k_targets = k_targets.loc[(k_targets['KINASES_ORFS'] != '-') | (k_targets['PHOSPHATASES_ORFS'] != '-')]

    k_targets['SOURCE'] = k_targets['KINASES_ORFS'] + '|' + k_targets['PHOSPHATASES_ORFS']
    k_targets = [(k, t + '_' + site) for t, site, source in k_targets[['ORF_NAME', 'PHOSPHO_SITE', 'SOURCE']].values for k in source.split('|') if k != '-' and k != '']
    k_targets = DataFrame(k_targets, columns=['kinase', 'site'])

    if remove_self:
        k_targets = k_targets[[source != target.split('_')[0] for source, target in k_targets.values]]

    k_targets['value'] = 1

    k_targets = pivot_table(k_targets, values='value', index='site', columns='kinase', fill_value=0)

    return k_targets


# -- Linear regression models functions


def regress_out(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))

    xs, ys = x[mask], y[mask]
    xs = xs[ys.index]

    lm = LinearRegression().fit(np.mat(xs).T, ys)
    ys_ = ys - lm.coef_[0] * xs - lm.intercept_
    return dict(zip(np.array(ys.index), ys_))


def estimate_activity_with_sklearn(x, y, alpha=.1):
    ys = y.dropna()
    xs = x.ix[ys.index].replace(np.NaN, 0.0)

    xs = xs.loc[:, xs.sum() != 0]

    lm = Ridge(fit_intercept=True, alpha=alpha).fit(xs, zscore(ys))

    return dict(zip(*(xs.columns, lm.coef_)))


def estimate_activity_with_statsmodel(x, y, L1_wt=0):
    ys = y.dropna()
    xs = x.ix[ys.index].replace(np.NaN, 0.0)

    xs = xs.loc[:, xs.sum() != 0]

    lm = sm.OLS(zscore(ys), st.add_constant(xs))

    res = lm.fit_regularized(L1_wt=L1_wt)

    return res.params.drop('const').to_dict()


# -- Protein related utility functions


def get_protein_sequence():
    return read_csv('%s/files/PhosphoGrid.txt' % wd, sep='\t').groupby('ORF_NAME')['SEQUENCE'].first().to_dict()


def get_site(protein, peptide):
    pep_start = protein.find(re.sub('\[[0-9]*\.?[0-9]*\]', '', peptide))
    pep_site_strat = peptide.find('[')
    site_pos = pep_start + pep_site_strat
    return protein[site_pos - 1] + str(site_pos)


def get_multiple_site(protein, peptide):
    n_sites = len(re.findall('\[[0-9]*\.?[0-9]*\]', peptide))
    return [get_site(protein, peptide if i == 0 else re.sub('\[[0-9]*\.?[0-9]*\]', '', peptide, i)) for i in xrange(n_sites)]


# -- Statistical utility functions


def pearson(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = pearsonr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue, mask.sum()


def spearman(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = spearmanr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue, mask.sum()


def cohend(x, y):
    return (np.mean(x) - np.mean(y)) / (np.sqrt((np.std(x) ** 2 + np.std(y) ** 2) / 2))


def metric(func, x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    res = func(x[mask], y[mask]) if np.sum(mask) > 5 else np.NaN
    return res if np.isfinite(res) else [[res]]


def shuffle(df):
    col, idx = df.columns, df.index
    val = df.values
    shape = val.shape
    val_flat = val.flatten()
    np.random.shuffle(val_flat)
    return DataFrame(val_flat.reshape(shape), columns=col, index=idx)


def count_percentage(df):
    return float(df.count().sum()) / np.prod(df.shape) * 100


# ---- Similarity scores

AA_PRIORS_YEAST = Series({
    'A': 0.055, 'R': 0.045, 'N': 0.061, 'D': 0.058, 'C': 0.013,
    'Q': 0.039, 'E': 0.064, 'G': 0.050, 'H': 0.022, 'I': 0.066,
    'L': 0.096, 'K': 0.073, 'M': 0.021, 'F': 0.045, 'P': 0.044,
    'S': 0.091, 'T': 0.059, 'W': 0.010, 'Y': 0.034, 'V': 0.056
})

AA_PRIORS_HUMAN = Series({
    'A': 0.070, 'R': 0.056, 'N': 0.036, 'D': 0.048, 'C': 0.023,
    'Q': 0.047, 'E': 0.071, 'G': 0.066, 'H': 0.026, 'I': 0.044,
    'L': 0.100, 'K': 0.058, 'M': 0.021, 'F': 0.037, 'P': 0.063,
    'S': 0.083, 'T': 0.053, 'W': 0.012, 'Y': 0.027, 'V': 0.060
})

namespace = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def read_fasta(fasta_file=None):
    sequences = {}

    with open(fasta_file) as f:
        lines = f.readlines()

        for i in range(len(lines)):
            if lines[i].startswith('>'):
                key = lines[i].split(' ')[0].split('>')[1].strip()
                sequence = ''

                i += 1
                while (i < len(lines)) and (not lines[i].startswith('>')):
                    sequence += lines[i].strip()
                    i += 1

                sequences[key] = sequence.replace('*', '')

    return sequences


def flanking_sequence(seqs, sites, flank=7, empty_char='-'):
    flanking_regions = {}

    for s in sites:
        tar, psite = s.split('_')
        res, pos = psite[0], int(psite[1:])
        sequence = seqs[tar]

        seq_pos = pos - 1

        if seq_pos > len(sequence):
            print '[WARNING] P-site (%s) placed outside protein sequence' % s
            continue

        start_pos = max(0, seq_pos - flank)
        end_pos = min(len(sequence), seq_pos + flank + 1)

        prefix = empty_char * (abs(seq_pos - flank) if seq_pos - flank < 0 else 0)
        sufix = empty_char * (abs((seq_pos + flank + 1) - len(sequence)) if (seq_pos + flank + 1) > len(sequence) else 0)

        flanking_region = prefix + sequence[start_pos: end_pos] + sufix

        flanking_regions[s] = flanking_region

    return flanking_regions


def position_weight_matrix(flanking_regions, priors, relative_freq=True):
    # Build matrix of flanking regions
    m = DataFrame([list(v) for v in flanking_regions.values()], index=flanking_regions.keys())

    # Construct Position Weighted Matrix (PWM)
    pwm_m = DataFrame({c: {aa: count for aa, count in zip(*(np.unique(m[c], return_counts=True)))} for c in m}).replace(np.NaN, 0)

    # Relative frequencies
    if relative_freq:
        pwm_m = pwm_m / pwm_m.sum()

    # Sort PWM matrix and priors
    pwm_m, priors = pwm_m.ix[namespace], priors[namespace]

    # Information content
    ic = (pwm_m * np.log2(pwm_m.divide(priors, axis=0))).sum()

    return pwm_m.replace(np.NaN, 0), ic


def score_sequence(sequence, pwm, empty_char='-'):
    return np.array([pwm.ix[sequence[i], i] if sequence[i] != empty_char else .0 for i in range(len(sequence))])


def similarity_score_matrix(flanking_regions, pwm, ic, ignore_central=True, is_kinase_pwm=True):
    central_ind = None

    # Only score sequences which have a central residue S/T or Y depending on the PWM
    kinase_type = '*'
    if is_kinase_pwm:
        kinase_type = 'S|T' if pwm.ix[:, int(np.floor(pwm.shape[1] / 2))].argmax() in 'S|T' else 'Y'

    # Central residue index
    if ignore_central:
        central_ind = int(np.floor(pwm.shape[1] / 2))

    # Best/worst sequence score
    bs = score_sequence(''.join(pwm.apply(lambda x: x.argmax())), pwm)
    ws = score_sequence(''.join(pwm.apply(lambda x: x.argmin())), pwm)

    # Which indicies do we keep
    positions_to_keep = list(pwm.drop(central_ind, axis=1)) if ignore_central else list(pwm)

    # Information content of positions to keep
    ic = ic[positions_to_keep]

    # Best and worst scores
    bs = (ic * bs[positions_to_keep]).sum()
    ws = (ic * ws[positions_to_keep]).sum()

    # Get array of scores
    sites_to_keep = [psite for psite, seq in flanking_regions.items() if seq[central_ind] in kinase_type] if is_kinase_pwm else flanking_regions.keys()

    # Score positions to keep
    sites_scores = {(site, flanking_regions[site]): score_sequence(flanking_regions[site], pwm)[positions_to_keep] for site in sites_to_keep}

    # Score p-sites flanking regions
    scores = Series({(site, seq): sum(ic * score) for (site, seq), score in sites_scores.items()})

    # Normalise scores
    scores = (scores - ws) / (bs - ws)

    return scores


def get_ko_strains():
    return set(read_csv('%s/files/steadystate_strains.txt' % wd, sep='\t')['strains'])


def get_proteins_name(uniprot_file='/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt'):
    return read_csv(uniprot_file, sep='\t', index_col=1)['gene'].to_dict()


def get_metabolites_name(annotation_file='%s/files/dynamic_metabolite_annotation.txt' % wd):
    annot = read_csv(annotation_file, sep='\t')
    annot['mz'] = ['%.4f' % i for i in annot['mz']]
    annot = annot.groupby('mz')['metabolite'].agg(lambda x: '; '.join(set(x))).to_dict()

    return annot


def get_metabolites_model_annot(annotation_file='%s/files/Annotation_Yeast_glucose.csv' % wd):
    annot = read_csv(annotation_file, sep=',')
    annot['mz'] = ['%.2f' % i for i in annot['mz']]

    counts = {mz: counts for mz, counts in zip(*(np.unique(annot['mz'], return_counts=True)))}
    annot = annot[[counts[i] == 1 for i in annot['mz']]]

    annot = annot.set_index('mz')

    return annot
