# MIT License
# 
# Copyright (c) 2021 Playtika Ltd.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np


def cohens_d(mu_1, mu_2, std):
    """

    Compute the standardized effect size as difference between the two means divided by the standard deviation.

    Parameters
    ----------
    mu_1 : float
        Mean of the first sample.
    mu_2 : float
        Mean of the second sample.
    std : float > 0
        Pooled standard deviation. It assumes that the variance of each population is the same.

    Returns
    -------
    effect_size : float
        Effect size as cohen's d coefficient
    """

    return (mu_1 - mu_2) / std


def cohens_h(p1, p2):
    """

    Compute the effect size as measure of distance between two proportions or probabilities. It is the difference
    between their arcsine transformations

    Parameters
    ----------
    p1 : float in interval (0,1)
        Proportion or probability of the first sample.
    p2 : float in interval (0,1)
        Proportion or probability of the second sample.

    Returns
    -------
    effect_size : float
        Effect size as cohen's h coefficient
    """

    return abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def pooled_std(sample1, sample2):
    """

    Compute pooled standard deviation between two samples.

    Parameters
    ----------
    sample1 : array_like
        Observation of first sample
    sample2 : array_like
        Observation of second sample

    Returns
    -------
    pooled_std : float > 0
        p-value for the test
    """
    # Compute the size of samples
    n1, n2 = len(sample1), len(sample2)

    # Compute the variance of the samples
    std1, std2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)

    # Compute the pooled standard deviation
    return np.sqrt(((n1 - 1) * std1 + (n2 - 1) * std2) / (n1 + n2 - 2))
