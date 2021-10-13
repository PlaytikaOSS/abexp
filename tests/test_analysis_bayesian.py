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

from abexp.core.analysis_bayesian import BayesianAnalyzer, BayesianGLMAnalyzer


def test_bayesian_compare_mean():
    """ Test compare means from statistics. """

    # Generate random samples
    np.random.seed(42)
    obs1 = np.random.normal(490, 200, 1000)
    obs2 = np.random.normal(500, 200, 1000)

    analyzer = BayesianAnalyzer()

    prob, lift, diff_means, ci = analyzer.compare_mean(obs_contr=obs1, obs_treat=obs2)

    assert (type(prob) == np.float64), 'Error prob'
    assert (type(lift) == np.float64), 'Error lift'
    assert (type(diff_means) == np.float64), 'Error diff_means'
    assert (type(ci[0]) == np.float64), 'Error'
    assert (type(ci[1]) == np.float64), 'Error'
    # TODO: check value


def test_bayesian_compare_conv():
    """ Test compare conversions from statistics. """

    # Results based on the online calculator at
    # https://marketing.dynamicyield.com/bayesian-calculator/ for probability
    # https://abtestguide.com/bayesian/ for lift

    data = {'conv1': [300, 500, 3000, 100, 700],
            'conv2': [370, 450, 3200, 100, 100],
            'nobs1': [700, 5000, 10000, 980, 980],
            'nobs2': [800, 5000, 10000, 980, 980],
            'prob': [0.9058, 0.0431, 0.9989, 0.50, 0],
            'lift': [0.0792, -0.10, 0.0667, 0, -0.857]}

    # Set epsilons
    eps = 0.005

    analyzer = BayesianAnalyzer()

    for i in np.arange(len(data['prob'])):
        prob, lift = analyzer.compare_conv(conv_contr=data['conv1'][i],
                                           conv_treat=data['conv2'][i],
                                           nobs_contr=data['nobs1'][i],
                                           nobs_treat=data['nobs2'][i])

        assert (data['prob'][i] - eps < prob < data['prob'][i] + eps), \
            'Error in compare_conv computed prob={}, expected_prob={}'.format(prob, data['prob'][i])

        assert (data['lift'][i] - eps < lift < data['lift'][i] + eps), \
            'Error in compare_conv computed lift={}, expected_lift={}'.format(lift, data['lift'][i])


def test_bayesian_multivariate_glm():
    """ Test bayesiann GLM with multivariate regression. """

    analyzer = BayesianGLMAnalyzer()

    df1 = pd.DataFrame([[1, 4, 35],
                        [0, 4, 5],
                        [1, 3, 28],
                        [0, 1, 5],
                        [0, 2, 1],
                        [1, 0, 1.5]], columns=['group', 'level', 'kpi'])

    df2 = pd.DataFrame([[0, 0, 100],
                        [0, 1, 100],
                        [0, 0, 100],
                        [1, 0, 100],
                        [1, 1, 100],
                        [1, 0, 100]], columns=['group', 'level', 'kpi'])

    stats = analyzer.multivariate_regression(df1, 'kpi')
    assert (type(stats) == pd.DataFrame), 'Error'
    # TODO: check value

    stats = analyzer.multivariate_regression(df2, 'kpi')
    assert (type(stats) == pd.DataFrame), 'Error'
    # TODO: check value


def test_bayesian_hierarchical_glm():
    """ Test bayesiann GLM with hierarchical regression. """

    analyzer = BayesianGLMAnalyzer()

    df1 = pd.DataFrame([[0, 5,   'Italy'],
                        [0, 5,   'Italy'],
                        [0, 100, 'Switzerland'],
                        [1, 100, 'Italy'],
                        [1, 100, 'Italy'],
                        [1, 100, 'France']], columns=['group', 'kpi', 'country'])

    stats = analyzer.hierarchical_regression(df1, group_col='group', cat_col='country', kpi_col='kpi')
    assert (type(stats) == pd.DataFrame), 'Error'
    # TODO: check value
