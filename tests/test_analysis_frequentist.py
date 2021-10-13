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
import pandas as pd
import statsmodels.api as sm

from abexp.core.analysis_frequentist import FrequentistAnalyzer


def test_compare_mean_obs():
    """ Test means comparisons from observations. """

    np.random.seed(42)
    obs1 = np.random.normal(500, 200, 1000)
    obs2 = np.random.normal(480, 200, 1000)
    obs3 = np.random.normal(90, 15, 1000)
    obs4 = np.random.normal(400, 200, 1000)

    # Results based on the online calculator at https://www.evanmiller.org/ab-testing/t-test.html
    data = {'obs1': [obs1, obs3, obs1],
            'obs2': [obs2, obs3, obs4],
            'p_val': [0.27, 1, 0],
            'ci_contr': [[491.713, 516.019], [89.462, 91.382], [491.713, 516.019]],
            'ci_treat': [[481.788, 506.547], [89.462, 91.382], [391.713, 416.019]]}

    # Set epsilons
    eps_pval = 0.003
    eps_ci = 0.05  #

    analyzer = FrequentistAnalyzer()

    for i in np.arange(len(data['p_val'])):

        p_val, ci_contr, ci_treat = analyzer.compare_mean_obs(obs_contr=data['obs1'][i], obs_treat=data['obs2'][i])

        assert (data['p_val'][i]-eps_pval < p_val < data['p_val'][i]+eps_pval), \
            'Error in compare_mean_obs computed p_val={}, expected_p_val={}'.format(p_val, data['p_val'][i])

        assert (data['ci_contr'][i][0]*(1-eps_ci) < ci_contr[0] < data['ci_contr'][i][0]*(1+eps_ci) and
                data['ci_contr'][i][1]*(1-eps_ci) < ci_contr[1] < data['ci_contr'][i][1]*(1+eps_ci)), \
            'Error in compare_mean_obs computed ci_contr={}, expected ci_contr={}'.format(ci_contr, data['ci_contr'][i])

        assert (data['ci_treat'][i][0]*(1-eps_ci) < ci_treat[0] < data['ci_treat'][i][0]*(1+eps_ci) and
                data['ci_treat'][i][1]*(1-eps_ci) < ci_treat[1] < data['ci_treat'][i][1]*(1+eps_ci)), \
            'Error in compare_mean_obs computed ci_treat={}, expected ci_treat={}'.format(ci_treat, data['ci_treat'][i])


def test_compare_mean_stats():
    """ Test means comparisons from statistics. """

    # Results based on the online calculator at https://www.evanmiller.org/ab-testing/t-test.html
    data = {'mu1': [495, 95, 1000, 50],
            'mu2': [500, 90, 980, 50],
            'std': [300, 20, 50, 5],
            'nobs1': [5000, 100, 200, 10000],
            'nobs2': [5000, 100, 200, 10000],
            'p_val': [0.4, 0.079, 0, 1],
            'ci_contr': [[486.683, 503.317], [91.032, 98.968], [993.028, 1006.972], [49.902, 50.098]],
            'ci_treat': [[491.683, 508.317], [86.032, 93.968], [973.028, 986.972], [49.902, 50.098]]}

    # Set epsilons
    eps_pval = 0.005
    eps_ci = 0.05

    analyzer = FrequentistAnalyzer()

    for i in np.arange(len(data['p_val'])):

        p_val, ci_contr, ci_treat = analyzer.compare_mean_stats(mean_contr=data['mu1'][i],
                                                                mean_treat=data['mu2'][i],
                                                                std_contr=data['std'][i],
                                                                nobs_contr=data['nobs1'][i],
                                                                nobs_treat=data['nobs2'][i])

        assert (data['p_val'][i]-eps_pval < p_val < data['p_val'][i]+eps_pval), \
            'Error in compare_mean_obs computed p_val={}, expected_p_val={}'.format(p_val, data['p_val'][i])

        assert (data['ci_contr'][i][0]*(1-eps_ci) < ci_contr[0] < data['ci_contr'][i][0]*(1+eps_ci) and
                data['ci_contr'][i][1]*(1-eps_ci) < ci_contr[1] < data['ci_contr'][i][1]*(1+eps_ci)), \
            'Error in compare_mean_obs computed ci_contr={}, expected ci_contr={}'.format(ci_contr, data['ci_contr'][i])

        assert (data['ci_treat'][i][0]*(1-eps_ci) < ci_treat[0] < data['ci_treat'][i][0]*(1+eps_ci) and
                data['ci_treat'][i][1]*(1-eps_ci) < ci_treat[1] < data['ci_treat'][i][1]*(1+eps_ci)), \
            'Error in compare_mean_obs computed ci_treat={}, expected ci_treat={}'.format(ci_treat, data['ci_treat'][i])


def test_compare_conv_stats():
    """ Test conversions comparisons from statistics. """

    # Results based on the online calculator at https://www.evanmiller.org/ab-testing/chi-squared.html
    data = {'conv1': [300, 450, 3000, 100, 700],
            'conv2': [370, 500, 3200, 100, 100],
            'nobs1': [700, 5000, 10000, 980, 980],
            'nobs2': [800, 5000, 10000, 980, 980],
            'p_val': [0.19, 0.088, 0.00223, 1, 0],
            'ci_contr': [[0.392, 0.466], [0.082, 0.098], [0.291, 0.309], [0.085, 0.123], [0.685, 0.742]],
            'ci_treat': [[0.428, 0.497], [0.092, 0.109], [0.311, 0.329], [0.085, 0.123], [0.085, 0.123]]}

    # Set epsilons
    eps_pval = 0.005
    eps_ci = 0.05

    analyzer = FrequentistAnalyzer()

    for i in np.arange(len(data['p_val'])):

        p_val, ci_contr, ci_treat = analyzer.compare_conv_stats(conv_contr=data['conv1'][i],
                                                                conv_treat=data['conv2'][i],
                                                                nobs_contr=data['nobs1'][i],
                                                                nobs_treat=data['nobs2'][i])

        assert (data['p_val'][i]-eps_pval < p_val < data['p_val'][i]+eps_pval), \
            'Error in compare_mean_obs computed p_val={}, expected_p_val={}'.format(p_val, data['p_val'][i])

        assert (data['ci_contr'][i][0]*(1-eps_ci) < ci_contr[0] < data['ci_contr'][i][0]*(1+eps_ci) and
                data['ci_contr'][i][1]*(1-eps_ci) < ci_contr[1] < data['ci_contr'][i][1]*(1+eps_ci)), \
            'Error in compare_mean_obs computed ci_contr={}, expected ci_contr={}'.format(ci_contr, data['ci_contr'][i])

        assert (data['ci_treat'][i][0]*(1-eps_ci) < ci_treat[0] < data['ci_treat'][i][0]*(1+eps_ci) and
                data['ci_treat'][i][1]*(1-eps_ci) < ci_treat[1] < data['ci_treat'][i][1]*(1+eps_ci)), \
            'Error in compare_mean_obs computed ci_treat={}, expected ci_treat={}'.format(ci_treat, data['ci_treat'][i])


def test_compare_conv_obs():
    """ Test conversions/proportions comparisons from observations. """

    obs1 = 200 * [1] + 200 * [0] + 200 * [1] + 200 * [0] + 4600 * [0]
    obs2 = 200 * [1] + 200 * [0] + 200 * [1] + 200 * [0] + 80 * [1] + 4520 * [0]

    # Results based on the online calculator at https://www.evanmiller.org/ab-testing/chi-squared.html
    data = {'obs1': [obs1],
            'obs2': [obs2],
            'p_val': [0.00489],
            'ci_contr': [[0.067, 0.081]],
            'ci_treat': [[0.082, 0.097]]}

    # Set epsilons
    eps_pval = 0.005
    eps_ci = 0.05

    analyzer = FrequentistAnalyzer()

    for i in np.arange(len(data['p_val'])):

        p_val, ci_contr, ci_treat = analyzer.compare_conv_obs(obs_contr=data['obs1'][i], obs_treat=data['obs2'][i])

        assert (data['p_val'][i]-eps_pval < p_val < data['p_val'][i]+eps_pval), \
            'Error in compare_mean_obs computed p_val={}, expected_p_val={}'.format(p_val, data['p_val'][i])

        assert (data['ci_contr'][i][0]*(1-eps_ci) < ci_contr[0] < data['ci_contr'][i][0]*(1+eps_ci) and
                data['ci_contr'][i][1]*(1-eps_ci) < ci_contr[1] < data['ci_contr'][i][1]*(1+eps_ci)), \
            'Error in compare_mean_obs computed ci_contr={}, expected ci_contr={}'.format(ci_contr, data['ci_contr'][i])

        assert (data['ci_treat'][i][0]*(1-eps_ci) < ci_treat[0] < data['ci_treat'][i][0]*(1+eps_ci) and
                data['ci_treat'][i][1]*(1-eps_ci) < ci_treat[1] < data['ci_treat'][i][1]*(1+eps_ci)), \
            'Error in compare_mean_obs computed ci_treat={}, expected ci_treat={}'.format(ci_treat, data['ci_treat'][i])


def test_check_homogeneity():
    """ Test homogeneity check. """

    # Get data for test
    spector_data = sm.datasets.spector.load_pandas()
    X = spector_data.exog.copy()
    y = spector_data.endog.copy()

    # Add categorical variables
    country = 5 * ['Italy'] + 15 * ['France'] + 10 * ['Switzerland'] + 2 * ['Israel']
    tier = 3 * [1] + 8 * [3] + 21 * [5]
    X['country'] = country
    X['tier'] = tier

    # Approximate values when comparing
    dec = 4

    analyzer = FrequentistAnalyzer()

    computed_stats = analyzer.check_homogeneity(X, y, cat_cols=['country', 'tier'])
    expected_stats = pd.read_csv('data/homogeneity_check.csv', index_col=0)

    comparison = np.all(np.around(expected_stats, dec) == np.around(computed_stats, dec))

    assert comparison, 'Error boostrap with std \n Expected stats: \n {} \n  Computed stats \n {} \n {}' \
        .format(expected_stats, computed_stats, comparison)


def test_bootstrap():
    """ Test bootstrapping. """

    # Generate random data
    np.random.seed(42)
    data = pd.DataFrame({'day1': np.random.randint(low=1, high=500, size=3),
                         'day2': np.random.randint(low=1, high=500, size=3),
                         'day3': np.random.randint(low=1, high=500, size=3),
                         'day4': np.random.randint(low=1, high=500, size=3),
                         'dayN': np.random.randint(low=1, high=500, size=3)})

    # Approximate values when comparing
    dec = 4

    analyzer = FrequentistAnalyzer()

    # Mean
    computed_mean = analyzer.bootstrap(data, func=np.mean, rep=7, seed=42)
    expected_mean = pd.read_csv('data/stats_boostrap_mean.csv', index_col=0)
    comparison_mean = np.all(np.around(expected_mean, dec) == np.around(computed_mean, dec))
    assert comparison_mean, 'Error boostrap with mean \n Expected stats: \n {} \n  Computed stats \n {} \n {}'\
        .format(expected_mean, computed_mean, comparison_mean)

    # Median
    computed_median = analyzer.bootstrap(data, func=lambda x: np.median(x, axis=0), rep=7, seed=42)
    expected_median = pd.read_csv('data/stats_boostrap_median.csv', index_col=0)
    comparison_median = np.all(np.around(expected_median, dec) == np.around(computed_median, dec))
    assert comparison_median, 'Error boostrap with median \n Expected stats: \n {} \n  Computed stats \n {} \n {}'\
        .format(computed_median, expected_median, comparison_median)

    # Sum
    computed_sum = analyzer.bootstrap(data, func=np.sum, rep=7, seed=42)
    expected_sum = pd.read_csv('data/stats_boostrap_sum.csv', index_col=0)
    comparison_sum = np.all(np.around(expected_sum, dec) == np.around(computed_sum, dec))
    assert comparison_sum, 'Error boostrap with sum \n Expected stats: \n {} \n  Computed stats \n {} \n {}'\
        .format(expected_sum, computed_sum, comparison_sum)

    # Standard Deviation
    computed_std = analyzer.bootstrap(data, func=np.std, rep=7, seed=42)
    expected_std = pd.read_csv('data/stats_boostrap_std.csv', index_col=0)
    comparison_std = np.all(np.around(expected_std, dec) == np.around(computed_std, dec))
    assert comparison_std, 'Error boostrap with std \n Expected stats: \n {} \n  Computed stats \n {} \n {}'\
        .format(expected_std, computed_std, comparison_std)
