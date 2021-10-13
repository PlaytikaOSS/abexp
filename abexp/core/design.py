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
from scipy.stats import ttest_ind
from statsmodels.stats.power import NormalIndPower, TTestIndPower

from abexp.statistics.stats_metrics import cohens_d, cohens_h


class SampleSize:
    """
    This class provides some utils to be used before running A/B test experiments. It includes minimum sample size
    determination, power calculation and effect size estimation. It handles both the case of means comparison and
    proportions comparison. Results are computed via power analysis with closed-form solution or simulation under the
    assumption that sample data are normally distributed.
    """
    def __init__(self):
        pass

    @staticmethod
    def ssd_mean(mean_contr, mean_treat, std_contr, alpha=0.05, power=0.8):
        """
        Sample size determination (SDD) to compare means. Compute the minimum sample size needed to run A/B test
        experiments. The result is computed via power analysis with closed-form solution t-test. Effect size estimation
        is calculated with cohen's d coefficient.

        Parameters
        ----------
        mean_contr : float
            Mean of the control group.
        mean_treat : float
            Mean of the treatment group.
        std_contr : float > 0
            Standard deviation of the control group. It assumes that the standard deviation of the control group is
            equal to the standard deviation of the treatment group.
        alpha : float in interval (0,1)
            Significance level, default 0.05. It is the probability of a type I error, that is wrong rejections if the
            null hypothesis is true.
        power : float in interval (0,1)
            Statistical power of the test, default 0.8. It is one minus the probability of a type II error. Power is
            the probability that the test correctly rejects the null hypothesis if the alternative hypothesis is true.

        Returns
        -------
        sample_size : int
            Minimum sample size per each group
        """

        # If the means are equals the function returns infinity
        if mean_contr == mean_treat:
            return np.Inf

        # Compute effect size as Cohen's d
        effect_size = cohens_d(mu_1=mean_contr, mu_2=mean_treat, std=std_contr)

        # Compute t-test to solve sample size
        analysis = TTestIndPower()
        sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, nobs1=None, ratio=1.,
                                           alternative='two-sided')

        return round(sample_size)

    @staticmethod
    def ssd_mean_sim(mean_contr, mean_treat, std_contr, alpha=0.05, power=0.8, sims=1000, start_size=100, step_size=0,
                     max_size=10000):
        """
        Sample size determination (SDD) to compare means with simulation. Compute the minimum sample size needed to run
        A/B test experiments. The result is computed via power analysis with simulation through t-test.

        Parameters
        ----------
        mean_contr : float
            Mean of the control group.
        mean_treat : float
            Mean of the treatment group.
        std_contr : float > 0
            Standard deviation of the control group. It assumes that the standard deviation of the control group is
            equal to the standard deviation of the treatment group.
        alpha : float in interval (0,1)
            Significance level, default 0.05. It is the probability of a type I error, that is wrong rejections if the
            Null Hypothesis is true.
        power : float in interval (0,1)
            Statistical Power of the test, default 0.8. It is one minus the probability of a type II error. Power is the
            probability that the test correctly rejects the Null Hypothesis if the Alternative Hypothesis is true.
        sims : int
            Number simulations, default 1000.
        start_size : int
            Initial sample size, default 100, used for the first iteration.
        step_size : int
            Spacing between samples size, default 50. This is the distance between two adjacent sample size,
            sample_size[i+1] - sample_size[i].
        max_size : int
            Maximum sample size, default 10000. The function returns this value if the desired power is not reached via
            simulation.

        Returns
        -------
        sample_size : int
            Minimum sample size per each group
        """

        # If the means are equals the function returns infinity
        if mean_contr == mean_treat:
            return np.Inf

        # Initialize simulated power and sample size, sample size
        sim_power, sample_size = 0, start_size - step_size

        while sim_power < power and sample_size <= max_size - step_size:

            # Keep incrementing sample size by step_size till the required power is reached
            sample_size += step_size

            # Model the variable for the two groups as normal distributions
            obs_contr = np.random.normal(loc=mean_contr, scale=std_contr, size=(sample_size, sims))
            obs_treat = np.random.normal(loc=mean_treat, scale=std_contr, size=(sample_size, sims))

            # Compute t-test for control and treatment groups
            _, p_value = ttest_ind(obs_treat, obs_contr)

            # Power is the fraction of times in the simulation when the p-value was less than 0.05
            sim_power = (p_value < alpha).sum() / sims

        return round(sample_size)

    @staticmethod
    def ssd_prop(prop_contr, prop_treat, alpha=0.05, power=0.8):
        """
        Sample size determination (SDD) to compare proportions. Compute the minimum sample size needed to run A/B test
        experiments. The result is computed via power analysis with closed-form solution z-test. Effect size
        estimation is calculated with cohen's h coefficient.

        Parameters
        ----------
        prop_contr : float in interval (0,1)
            Proportion in the control group.
        prop_treat : float in interval (0,1)
            Proportion in the treatment group.
        alpha : float in interval (0,1)
            Significance level, default 0.05. It is the probability of a type I error, that is wrong rejections if the
            Null Hypothesis is true.
        power : float in interval (0,1)
            Statistical Power of the test, default 0.8. It is one minus the probability of a type II error. Power is
            the probability that the test correctly rejects the Null Hypothesis if the Alternative Hypothesis is true.

        Returns
        -------
        sample_size : int
            Minimum sample size per each group
        """

        # If proportions are equals the function returns infinity
        if prop_contr == prop_treat:
            return np.Inf

        # Compute effect size as Cohen's h
        effect_size = cohens_h(p1=prop_contr, p2=prop_treat)

        # Compute t-test to solve sample size
        analysis = NormalIndPower()
        sample_size = analysis.solve_power(effect_size=effect_size, nobs1=None, alpha=alpha, power=power, ratio=1.,
                                           alternative='two-sided')
        return round(sample_size)
