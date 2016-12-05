# -*- coding: utf-8 -*-
"""
============================
Partial conjunction analysis
============================

In this example we replicate Figure 2 from Bejamini et al. 2008 [1]_:

    In each of 1000 locations 10 independent unit variance Gaussian noise
    measurements were simulated, and in 100 locations a signal of
    size :math:`\mu` was added in k out of the 10 repetitions (k = 3, 7, 9)
    per location. The signal size :math:`\mu` was independently sampled for
    each location and map from a :math:`N(\mu_0,\sigma_0^2)
    distribution, where we varied :math:`\mu_0 = 2,...,6` and
    :math:`\sigma_0 = 0,...,\max(2,\mu_2/2)`.

    We pooled the p-values using [the Simes method], [the Stouffer method],
    or [the Fisher method], as well as using the maximum p-value method
    (see Friston et al. [2005] for details) because this is the only
    method used up to now for 1 < u < n. Next, we computed the resulting
    map threshold using the suggested BH procedure.

In their paper they only show the :math:`\sigma_0=2` case, so we replicate
that here, and for simplicity we limit ourselves to the Fisher, Stouffer,
and Simes methods of p-value combination across subjects.
"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from mne.stats import partial_conjunction

rng = np.random.RandomState(0)
n_subjects, n_locations, n_nz = 10, 1000, 100
subject_counts = (3, 7, 9)
methods = ('fisher', 'stouffer', 'simes')

unit_noise = rng.randn(n_subjects, n_locations)
sigma_0 = 2.
colspans = [2, 3, 4]
cis = [0, 2, 5]
alpha = 0.05
fig = plt.figure(figsize=(8, 5))
linestyles = dict(fisher='-', stouffer='--', simes=':')
for mi, mu_0 in enumerate((2, 4, 6)):
    for si, n_sig in enumerate(subject_counts):
        print(u'Running μ0=%s, %s signals' % (mu_0, n_sig))
        data = unit_noise.copy()
        data[:n_sig, :n_nz] += mu_0 + rng.randn(n_sig, n_nz) * sigma_0
        p = norm.cdf(-np.abs(data)) * 2
        del data

        ax = plt.subplot2grid((3, 9), (mi, cis[si]), colspan=colspans[si])
        for method in methods:
            rejected, p_conj = partial_conjunction(p, method=method)
            power = (p_conj[:n_sig, :n_nz] < alpha).mean(axis=-1)
            xs = np.arange(n_sig) + 1
            ax.plot(xs, power, linestyle=linestyles[method], color='k',
                    linewidth=0.5)
        ax.set(yticks=np.arange(6) / 5., xticks=xs)
        ax.set(xlim=[1, n_sig], ylim=[0, 1.1])
        if mi == 0:
            ax.set(title='k=%s' % n_sig)
        elif mi == 2:
            ax.set(xlabel='u')
        if si == 0:
            ax.set(ylabel='$\mu_0=%s$, $\sigma_0=%s$\nPower' % (mu_0, sigma_0))
fig.tight_layout()
plt.show()

###############################################################################
# References
# ----------
# .. [1] Heller R, Golland Y, Malach R, Benjamini Y (2007).
#    Conjunction group analysis: An alternative to mixed/random effect
#    analysis. NeuroImage 37:1178–1185
# .. [2] Benjamini Y, Heller R (2008). Screening for Partial Conjunction
#    Hypotheses. Biometrics 64:1215–1222
#
