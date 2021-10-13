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
from scipy.stats import normaltest, shapiro


def permutation_test(obs_1, obs_2, reps=1000) -> float:
    """
    Run the permutation test, a type of statistical significance test based on resampling method. The distribution
    of the test statistic under the null hypothesis is obtained by calculating all possible values of the test
    statistic under N possible rearrangements of the observed data points randomly selected. N is the number of
    repetitions.

    Parameters
    ----------
    obs_1 : array_like
        Observation of first sample
    obs_2 : array_like
        Observation of second sample
    reps : int > 0
        Number of repetition of the permutations

    Returns
    -------
    p_val : float in interval (0,1)
        p-value for the test
    """

    data = np.concatenate([obs_1, obs_2])

    # Create a pool of possible permutations whose size is equal to the number of repetitions
    perm = np.array([np.random.permutation(len(obs_1) + len(obs_2)) for _ in range(reps)])
    perm_1_datasets = data[perm[:, :len(obs_1)]]
    perm_2_datasets = data[perm[:, len(obs_1):]]

    # Calculate the difference in means for each of the datasets
    samples = np.mean(perm_1_datasets, axis=1) - np.mean(perm_2_datasets, axis=1)

    # Calculate the test statistic and p-value
    test_stat = np.mean(obs_1) - np.mean(obs_2)
    p_val = 2 * np.sum(np.abs(samples) >= np.abs(test_stat)) / reps

    return p_val


def normal_test(x, method='dagostino') -> float:
    """
    Perform a normality test for the sample data. It tests if the sample comes from a normal distribution.

    Parameters
    ----------
    x : array_like
        The array of sample data
    method : string
        statistical method to perform the normality test, default 'dagostino'. It value can be either 'dagostino'
        to perform the D’Agostino and Pearson’s test which combines skew and kurtosis or 'shapiro' to perform the
        Shapiro-Wilk test for normality. For N > 5000 the p-value with 'shapiro' may not be accurate.

    Returns
    -------
    p_val : float in interval (0,1)
        p-value for the test
    """

    p_val = np.nan
    if method == 'dagostino':
        _, p_val = normaltest(x)
    elif method == 'shapiro':
        _, p_val = shapiro(x)

    return p_val
