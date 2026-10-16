"""Agreement statistics for paired measurements.

Two measures are used throughout the scorecard:

* **Bland-Altman** limits of agreement (Bland & Altman 1986): the mean
  difference between two methods (the bias) and the band in which 95% of
  differences are expected to fall.
* **ICC(2,1)**, the intraclass correlation for a two-way random-effects
  model, absolute agreement, single measurement (Shrout & Fleiss 1979, case 2;
  McGraw & Wong 1996, ICC(A,1)). It is the right ICC when both raters or
  methods are taken as a sample of the ones that could have been used and a
  constant offset between them should count against agreement.

Pairs where either value is missing (NaN or infinite) are dropped before any
statistic is computed. Statistics that are undefined for the data at hand
(for example an SD from one pair) are returned as NaN without a warning.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Two-sided 95% quantile of the standard normal used for the limits of agreement.
LOA_Z = 1.96
# Spread below this fraction of the largest |value| is treated as rounding noise.
_RELATIVE_NOISE = 1e-9


def _paired(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same length, got {a.size} and {b.size}")
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def bland_altman(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Bias and 95% limits of agreement between two paired measurements.

    With differences :math:`d_i = a_i - b_i` over the :math:`n` complete pairs:

    * ``bias`` :math:`= \\bar d`
    * ``sd`` :math:`= \\sqrt{\\sum (d_i - \\bar d)^2 / (n - 1)}` (sample SD)
    * ``loa_low``, ``loa_high`` :math:`= \\bar d \\mp 1.96\\,\\mathrm{sd}`

    Parameters
    ----------
    a, b:
        Paired values of equal length, e.g. detector and reference times.
        ``a`` is the method under test, so a positive bias means ``a`` reads high (late).

    Returns
    -------
    dict
        ``bias``, ``sd``, ``loa_low``, ``loa_high`` (floats, NaN where
        undefined: no pairs for ``bias``, fewer than two for the rest) and
        ``n`` (int, number of complete pairs).
    """
    a, b = _paired(a, b)
    d = a - b
    n = int(d.size)
    bias = float(d.mean()) if n else float("nan")
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    return {
        "bias": bias,
        "sd": sd,
        "loa_low": bias - LOA_Z * sd,
        "loa_high": bias + LOA_Z * sd,
        "n": n,
    }


def icc_2_1_matrix(ratings: Sequence[Sequence[float]] | np.ndarray) -> float:
    """ICC(2,1) for an ``(n_subjects, k_raters)`` table of ratings.

    Two-way random effects, absolute agreement, single measurement
    (Shrout & Fleiss 1979, eq. for ICC(2,1); McGraw & Wong 1996, ICC(A,1)).
    From the two-way ANOVA without replication, with grand mean
    :math:`\\bar x`, row (subject) means :math:`\\bar x_{i\\cdot}` and column
    (rater) means :math:`\\bar x_{\\cdot j}`:

    * :math:`MS_R = k \\sum_i (\\bar x_{i\\cdot} - \\bar x)^2 / (n - 1)` (between subjects)
    * :math:`MS_C = n \\sum_j (\\bar x_{\\cdot j} - \\bar x)^2 / (k - 1)` (between raters)
    * :math:`MS_E = (SS_T - SS_R - SS_C) / ((n - 1)(k - 1))` (residual)

    .. math::

        ICC(2,1) = \\frac{MS_R - MS_E}{MS_R + (k - 1) MS_E + k (MS_C - MS_E) / n}

    Rows with any missing value are dropped. On Shrout & Fleiss's Table 2
    (6 targets, 4 judges) this gives 0.29, their published value.

    Parameters
    ----------
    ratings:
        One row per subject (e.g. per gait cycle), one column per rater or method.

    Returns
    -------
    float
        The ICC, or NaN when it is undefined: fewer than two complete subjects,
        fewer than two raters, or no variation beyond floating-point rounding
        (spread below 1e-9 of the largest ``|value|``).
    """
    x = np.asarray(ratings, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"ratings must be a 2-D (subjects, raters) array, got shape {x.shape}")
    x = x[np.all(np.isfinite(x), axis=1)]
    n, k = x.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = x.mean()
    ss_total = float(np.sum((x - grand) ** 2))
    # Values that agree up to floating-point rounding (e.g. a stride time of
    # 1.05 s computed as differences of event times) have no real variance;
    # the ratio of two rounding errors would be an arbitrary number.
    if ss_total <= x.size * (_RELATIVE_NOISE * float(np.max(np.abs(x)))) ** 2:
        return float("nan")
    ss_rows = k * float(np.sum((x.mean(axis=1) - grand) ** 2))
    ss_cols = n * float(np.sum((x.mean(axis=0) - grand) ** 2))
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    # Rounding can push a perfect-agreement residual a hair below zero.
    ms_err = max(ss_total - ss_rows - ss_cols, 0.0) / ((n - 1) * (k - 1))
    # Non-negative for n, k >= 2; zero only without any variation.
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    if denom <= 0:
        return float("nan")
    return float((ms_rows - ms_err) / denom)


def icc_2_1(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """ICC(2,1) between two raters or methods measuring the same subjects.

    Same as :func:`icc_2_1_matrix` with ``k = 2`` columns ``[a, b]``; see there
    for the formula. A constant offset between ``a`` and ``b`` lowers the
    value (absolute agreement), unlike a consistency ICC.

    Parameters
    ----------
    a, b:
        Paired values of equal length. Pairs with a missing value are dropped.

    Returns
    -------
    float
        The ICC, 1.0 for identical inputs that vary between subjects, NaN
        when undefined (fewer than two complete pairs or no variation).
    """
    a, b = _paired(a, b)
    return icc_2_1_matrix(np.column_stack([a, b]))
