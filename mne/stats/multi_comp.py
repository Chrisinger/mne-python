# -*- coding: utf-8 -*-
# Authors: Josef Pktd and example from H Raja and rewrite from Vincent Davis
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# Code borrowed from statsmodels
#
# License: BSD (3-clause)

import numpy as np
from ..externals.six import string_types


def _ecdf(x):
    """No frills empirical cdf used in fdrcorrection."""
    nobs = len(x)
    return np.arange(1, nobs + 1) / float(nobs)


def fdr_correction(pvals, alpha=0.05, method='indep'):
    """P-value correction with False Discovery Rate (FDR).

    Correction for multiple comparisons using FDR [1]_, [2]_.

    Parameters
    ----------
    pvals : array_like
        Set of p-values of the individual tests.
    alpha : float
        Error rate.
    method : 'indep' | 'negcorr'
        If 'indep' use Benjamini/Hochberg [1]_ for independent or
        positively correlated data (usually reasonable for neuroimaging),
        or 'negcorr' for Benjamini/Yekutieli [2]_.

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not.
    pval_corrected : array
        p-values adjusted for multiple hypothesis testing to limit FDR.

    References
    ----------
    .. [1] Benjamini Y, Hochberg Y (1995). Controlling the
       False Discovery Rate: A Practical and Powerful Approach to Multiple
       Testing. Journal of the Royal Statistical Society.
       Series B (Methodological) 57:289-300
    .. [2] Benjamini Y, Yekutieli D (2001) The Control of the False
       Discovery Rate in Multiple Testing under Dependency.
       The Annals of Statistics 29:1165–1188
    """
    pvals = np.asarray(pvals)
    shape_init = pvals.shape
    pvals = pvals.ravel()

    pvals_sortind = np.argsort(pvals)
    pvals_sorted = pvals[pvals_sortind]
    sortrevind = pvals_sortind.argsort()

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1. / np.arange(1, len(pvals_sorted) + 1))
        ecdffactor = _ecdf(pvals_sorted) / cm
    else:
        raise ValueError("Method should be 'indep' and 'negcorr'")

    reject = pvals_sorted < (ecdffactor * alpha)
    if reject.any():
        reject[:np.nonzero(reject)[0][-1] + 1] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    pvals_corrected[pvals_corrected > 1.0] = 1.0
    pvals_corrected = pvals_corrected[sortrevind].reshape(shape_init)
    reject = reject[sortrevind].reshape(shape_init)
    return reject, pvals_corrected


def bonferroni_correction(pval, alpha=0.05):
    """P-value correction with Bonferroni method.

    Parameters
    ----------
    pval : array_like
        Set of p-values of the individual tests.
    alpha : float
        Error rate.

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not
    pval_corrected : array
        pvalues adjusted for multiple hypothesis testing to limit FDR
    """
    pval = np.asarray(pval)
    pval_corrected = pval * float(pval.size)
    pval_corrected[pval_corrected > 1.] = 1.
    reject = pval_corrected < alpha
    return reject, pval_corrected


def partial_conjunction(p, alpha=0.05, method='fisher', fdr_method='indep'):
    """Compute the partial conjunction map.

    Parameters
    ----------
    p : ndarray, shape (n_subjects, ...)
        The p-value maps for each subject at each location ``(...)``.
    alpha : float
        The FDR correction threshold for each conjunction map.
    method : str
        The method used to combine the p values, can be "fisher" (default)
        or "stouffer", both of which are valid for independent
        (across subjects) p-values only, and "simes", which should be
        suitable for dependent p-values.
    fdr_method : str
        If 'indep' it implements Benjamini/Hochberg for independent or if
        'negcorr' it corresponds to Benjamini/Yekutieli (across locations).

    Returns
    -------
    rejected : ndarray, shape (...)
        The largest number of subjects for which the null hypothesis
        was rejected at each location.
    p : ndarray, shape (n_subjects, ...)
        The group-corrected p-values for each subject count (first
        dimension) at each location.

    Notes
    -----
    Quoting [1]_:

        Let :math:`k` be the (unknown) number of conditions or subjects
        that show real effect. The problem of testing in every brain
        voxel :math:`v` whether at least :math:`u` out of :math:`n`
        conditions or subjects considered show real effects, can be
        generally stated as follows:

        .. math::

            H_{0v}^{u/n}: k<u\ \textrm{versus}\ H_{1v}^{u/n}: k \geq u

        We shall call :math:`H_{0v}^{u/n}` the partial conjunction
        null hypothesis.

    And note from [2]_:

        The (perhaps more) intuitive procedure in such settings, to apply
        an FDR controlling procedure on each p-value map separately and then
        take the intersection of the discovered locations, does not control
        the FDR of the combined discoveries.

    References
    ----------
    .. [1] Heller R, Golland Y, Malach R, Benjamini Y (2007).
       Conjunction group analysis: An alternative to mixed/random effect
       analysis. NeuroImage 37:1178–1185
       https://dx.doi.org/10.1016/j.neuroimage.2007.05.051
    .. [2] Benjamini Y, Heller R (2008). Screening for Partial Conjunction
       Hypotheses. Biometrics 64:1215–1222
       https://doi.org/10.1111/j.1541-0420.2007.00984.x
    """
    from scipy.stats import combine_pvalues
    known_types = ('fisher', 'stouffer', 'simes')
    if not isinstance(method, string_types) or method not in known_types:
        raise ValueError('Method must be one of %s, got %s'
                         % (known_types, method))
    p = np.array(p)  # copy
    bad = ((p <= 0) | (p > 1)).sum()
    if bad > 0:
        raise ValueError('All p-values must be positive and at most 1, got %s '
                         'invalid values' % (bad,))
    orig_shape = p.shape
    # Sort the p-values after effectively reshaping to (-1, n_subjects)
    p = np.reshape(p, (len(p), -1)).T
    p_sort = np.sort(p)
    # At each location
    for pi, pp in enumerate(p):
        # For each hypothesis count
        for ii in range(p.shape[1]):
            # combine the n - u + 1 largest p-values, and
            # when n=u this trivially collapses to the last (largest) value
            # (NB. ii = u - 1)
            # ii = 0 is the global null (should combine all p values), and
            # ii = p.shape[1] - 1 should use the maximum p value
            if method == 'simes':
                n = p.shape[1] - ii
                p[pi, ii] = n * (p_sort[pi, ii:] / np.arange(1, n + 1)).min()
            else:
                p[pi, ii] = combine_pvalues(p_sort[pi, ii:], method=method)[1]
    # FDR correct each map
    for ii in range(p.shape[1]):
        p[:, ii] = fdr_correction(p[:, ii], method=fdr_method)[1]
    # Get the subject count
    rejected = np.reshape((p < alpha).sum(-1), orig_shape[1:])
    p = np.reshape(p.T, orig_shape)
    return rejected, p
