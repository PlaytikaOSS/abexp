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

from abexp.core.design import SampleSize


class Planning:

    @staticmethod
    def planning_diff_mean(avg_n_users_per_day, mean_contr, mean_treat, std_contr, alpha=0.05, power=0.8):
        """
        Use the sample size determination with means comparison from the core.design.SampleSize class
        to estimate the number of days that a test must run to achieve the desired significance and power level.

        Parameters
        ----------
        avg_n_users_per_day : int
            The number users per day which can be directed to the variant.
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
        n_days : int
            Minimum number of days to run the A/B test.
        """

        # If the means are equals the function returns infinity
        if mean_contr == mean_treat:
            return np.Inf

        # If the number of users per day is invalid returns infinity
        if avg_n_users_per_day <= 0:
            return np.Inf

        # Compute required sample size
        sample_size = SampleSize.ssd_mean(mean_contr, mean_treat, std_contr, alpha, power)

        return int(np.ceil(sample_size/avg_n_users_per_day))

    @staticmethod
    def planning_diff_prop(avg_n_users_per_day, prop_contr, prop_treat, alpha=0.05, power=0.8):
        """
        Use the sample size determination with proportions comparison from the core.design.SampleSize class
        to estimate the number of days that a test must run to achieve the desired significance and power level.

        Parameters
        ----------
        avg_n_users_per_day : int
            The number users per day which can be directed to the variant.
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
        n_days : int
            Minimum number of days to run the A/B test.
        """

        # If the means are equals the function returns infinity
        if prop_contr == prop_treat:
            return np.Inf

        # If the number of users per day is invalid returns infinity
        if avg_n_users_per_day <= 0:
            return np.Inf

        # Compute required sample size
        sample_size = SampleSize.ssd_prop(prop_contr, prop_treat, alpha, power)

        return int(np.ceil(sample_size / avg_n_users_per_day))
