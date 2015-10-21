from __future__ import division
import numpy as np
from pandas import DataFrame, Series, read_csv


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
