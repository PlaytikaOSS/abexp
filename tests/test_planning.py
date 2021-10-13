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

from abexp.core.planning import Planning


def test_planning_diff_equal_means():
    """
    If means are equal the result is expected to be infinity
    """

    # expected result is set to infinity
    res = np.Inf

    # means values are both set to a random float number
    test_values = [0, 2, 10, 100.8, 250, 375.95, 500.2, 1000, 1500.35, 3000, 10000]

    for v in test_values:
        s = Planning.planning_diff_mean(avg_n_users_per_day=500, mean_contr=v, mean_treat=v, std_contr=100)
        assert s == res, 'Error in planning when means are equals; computed_nb_days={}, ' \
                         'expected_nb_days={}'.format(s, res)


def test_planning_diff_mean_invalid_nb_users():
    """
    If the average number of users is invalid, should return infinity
    """

    # expected result is set to infinity
    res = np.Inf

    # means values are both set to a random float number
    test_values = [0, -5, -100]

    for v in test_values:
        s = Planning.planning_diff_mean(avg_n_users_per_day=v, mean_contr=20, mean_treat=30, std_contr=100)
        assert s == res, 'Error in planning when number of daily users is invalid; computed_nb_days={}, ' \
                         'expected_nb_days={}'.format(s, res)


def test_planning_diff_prop_equal_input():
    """
    If proportions are equal the result is expected to be infinity
    """

    # expected result is set to infinity
    res = np.Inf

    # means values are both set to a random float number
    test_values = [0, 2, 10, 100.8, 250, 375.95, 500.2, 1000, 1500.35, 3000, 10000]

    for v in test_values:
        s = Planning.planning_diff_prop(avg_n_users_per_day=500, prop_contr=v, prop_treat=v)
        assert s == res, 'Error in planning when proportions are equals; computed_nb_days={}, ' \
                         'expected_nb_days={}'.format(s, res)


def test_planning_diff_prop_invalid_nb_users():
    """
    If the average number of users is invalid, should return infinity
    """

    # expected result is set to infinity
    res = np.Inf

    # means values are both set to a random float number
    test_values = [0, -5, -100]

    for v in test_values:
        s = Planning.planning_diff_prop(avg_n_users_per_day=v, prop_contr=0.2, prop_treat=0.3)
        assert s == res, 'Error in planning when number of daily users is invalid; computed_nb_days={}, ' \
                         'expected_nb_days={}'.format(s, res)


def test_planning_diff_mean_pos():
    """
    Compare to online calculators the result of Sample Size Determination for means. Due to approximations,
    the test passes if sample_size belongs to the right neighbourhood of the result given by the online calculator;
    in other words the test passes if sample_size is in range (online_res, online_res + epsilon)
    """

    # Results based on the online calculator at https://www.stat.ubc.ca/~rollin/stats/ssize/n2.html
    data = {'mu_1': [10, 322, 1000, 5000, 300, 70000, -10, -542, -4000, 50],
            'mu_2': [9, 321, 973, 6200, 298, 68000, -9, -538, -4200, -42],
            'std': [1, 50, 75, 4000, 50, 40000, 1, 90, 1500, 500],
            'res': [16, 39245, 122, 175, 9812, 6280, 16, 7947, 883, 464]}

    # Test for different number of users per day
    users_per_day_list = [4, 100, 1500]
    # Set epsilon to 1
    eps = 1

    for users_per_day in users_per_day_list:
        for mu_1, mu_2, std, res in zip(data['mu_1'], data['mu_2'], data['std'], data['res']):
            estimated_nb_days = Planning.planning_diff_mean(users_per_day, mean_contr=mu_1, mean_treat=mu_2,
                                                            std_contr=std)
            expected = int(np.ceil(res / users_per_day))
            assert estimated_nb_days in range(expected, expected + eps + 1), \
                'Error in planning with avg_users_per_day={} m1={} mu_2={} std={}; computed_nb_days={}, ' \
                'expected_nb_days={}'.format(users_per_day, mu_1, mu_2, std, estimated_nb_days, expected)


def test_planning_diff_prop_pos():
    """
    Compare to online calculators the result of Sample Size Determination for prop difference. Due to approximations,
    the test passes if sample_size belongs to the neighbourhood of the result given by the online calculator;
    in other words the test passes if sample_size is in range (online_res - epsilon, online_res + epsilon)
    """

    # Results based on the online calculator at https://www.stat.ubc.ca/~rollin/stats/ssize/b2.html
    data = {'p1': [0.50, 0.80, 0.33, 0.80, 0.04, 0.12, 0.25, 0.40, 0.63, 0.09],
            'p2': [0.47, 0.81, 0.31, 0.40, 0.03, 0.11, 0.22, 0.42, 0.64, 0.10],
            'res': [4356, 24641, 8539, 23, 5301, 15976, 3135, 9493, 36383, 13495]}

    # Test for different number of users per day
    users_per_day_list = [4, 100, 1500]

    # Set epsilon to 0.5% of the result from the online calculator
    eps_list = [(r * 0.005 + 1) for r in data['res']]

    for users_per_day in users_per_day_list:
        for p1, p2, res, eps in zip(data['p1'], data['p2'], data['res'], eps_list):
            estimated_nb_days = Planning.planning_diff_prop(users_per_day, prop_contr=p1, prop_treat=p2)
            expected = int(np.ceil(res / users_per_day))
            assert estimated_nb_days in range(expected - int(eps), expected + int(eps) + 1), \
                'Error in planning with prop with p1={} p2={}; ' \
                'computed_nb_days={} ' \
                'expected_nb_days={}'.format(p1, p2, estimated_nb_days, expected)
