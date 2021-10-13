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
from scipy.stats import sem, t, ttest_ind, ttest_ind_from_stats
from sklearn.utils import resample
from statsmodels.api import add_constant
from statsmodels.formula.api import logit
from statsmodels.stats.proportion import proportion_confint, proportions_ztest


class FrequentistAnalyzer:
    """
    This class provides tools to perform analysis after A/B test experiments with frequentist statistical approach. It
    handles both the case of means comparison and conversions comparison with closed-form-solutions. It also includes
    bootstrapping and homogeneity checks of the observed samples.
    """
    def __init__(self):
        pass

    def compare_mean_stats(self, mean_contr, mean_treat, std_contr, nobs_contr,
                           nobs_treat, alpha=0.05):
        """
        Compare means from statistics. Compare the mean of the control group versus the mean of the treatment group.
        The result is computed with t-test (closed-form solution) given the groups statistics. It assumes that sample
        data are normally distributed.

        Parameters
        ----------
        mean_contr : float
            Mean of the control group.
        mean_treat : float
            Mean of the treatment group.
        std_contr : float > 0
            Standard deviation of the control group. It assumes that control and treatment group have the same standard
            deviation.
        nobs_contr : int > 0
            Number of observations in the control group
        nobs_treat : int > 0
            Number of observations in the treatment group
        alpha : float in interval (0,1)
            Significance level, default 0.05. It is the probability of a type I error, that is wrong rejections if the
            Null Hypothesis is true.

        Returns
        -------
        p_value : float in interval (0,1)
            p-value for the statistical test.
        ci_contr : tuple
            confidence interval for the control group.
        ci_treat : tuple
            confidence interval for the treatment group.
        """

        # Compute p-value via t-test
        _, p_value = ttest_ind_from_stats(mean1=mean_contr, std1=std_contr, nobs1=nobs_contr,
                                          mean2=mean_treat, std2=std_contr, nobs2=nobs_treat)

        # Define confidence level, degrees of freedom and standard error
        confidence_level = 1 - alpha
        df_contr = nobs_contr - 1
        df_treat = nobs_treat - 1
        se = std_contr * np.sqrt(1/nobs_contr + 1/nobs_treat)

        # Compute confidence intervals
        ci_contr = t.interval(confidence_level, df_contr, mean_contr, se)
        ci_treat = t.interval(confidence_level, df_treat, mean_treat, se)

        return p_value, ci_contr, ci_treat

    def compare_mean_obs(self, obs_contr, obs_treat, alpha=0.05):
        """
        Compare means from observed samples. Compare the mean of the control group versus the mean of the treatment
        group. The result is computed with t-test (closed-form solution) given the observed samples of the two groups.
        It assumes that sample data are normally distributed.

        Parameters
        ----------
        obs_contr : array_like
            Observation of the control sample. It contains the value to be analyzed per each sample.
        obs_treat : array_like
            Observation of the treatment sample. It contains the value to be analyzed per each sample.
        alpha : float in interval (0,1)
            Significance level, default 0.05. It is the probability of a type I error, that is wrong rejections if the
            Null Hypothesis is true.

        Returns
        -------
        p_value : float in interval (0,1)
            p-value for the statistical test.
        ci_contr : tuple
            confidence interval for the control group.
        ci_treat : tuple
            confidence interval for the treatment group.
        """

        # Compute p_value via t-test
        _, p_value = ttest_ind(obs_contr, obs_treat)

        # Define confidence level and degrees of freedom
        confidence_level = 1 - alpha
        df_contr = len(obs_contr) - 1
        df_treat = len(obs_treat) - 1

        # Compute confidence intervals
        ci_contr = t.interval(confidence_level, df_contr, np.mean(obs_contr), sem(obs_contr))
        ci_treat = t.interval(confidence_level, df_treat, np.mean(obs_treat), sem(obs_treat))

        return p_value, ci_contr, ci_treat

    def compare_conv_stats(self, conv_contr, conv_treat, nobs_contr, nobs_treat, alpha=0.05):
        """
        Compare conversions from statistics. Compare the conversions of the control group versus the conversions of the
        treatment group. The result is computed with z-test (closed-form solution) given the groups statistics. It
        assumes that sample data are normally distributed.

        Parameters
        ----------
        conv_contr : int > 0
            Number of conversions in the control group.
        conv_treat : int > 0
            Number of conversions in the treatment group.
        nobs_contr : int > 0
            Total number of observations of the control group.
        nobs_treat : int > 0
            Total number of observations of the treatment group.
        alpha : float in interval (0,1)
            Significance level, default 0.05. It is the probability of a type I error, that is wrong rejections if the
            Null Hypothesis is true.

        Returns
        -------
        p_value : float in interval (0,1)
            p-value for the statistical test.
        ci_contr : tuple
            confidence interval for the control group.
        ci_treat : tuple
            confidence interval for the treatment group.
        """

        # Rearrange input for z-test
        conv = np.array([conv_contr, conv_treat])
        nobs = np.array([nobs_contr, nobs_treat])

        # Compute p_value via z-test
        _, p_value = proportions_ztest(conv, nobs, alternative='two-sided')

        # Compute confidence intervals
        ci_low, ci_upp = proportion_confint(conv, nobs, alpha=alpha)
        ci_contr = [ci_low[0], ci_upp[0]]
        ci_treat = [ci_low[1], ci_upp[1]]

        return p_value, ci_contr, ci_treat

    def compare_conv_obs(self, obs_contr, obs_treat, alpha=0.05):
        """
        Compare conversions from observed samples. Compare the conversions of the control group versus the conversions
        of the treatment group. The result is computed with z-test (closed-form solution) given the observed samples of
        the two groups. It assumes that sample data are normally distributed.

        Parameters
        ----------
        obs_contr : array_like
            Observation of the control sample. It is a boolean vector (0 or 1) which indicates weather the sample i-th
            of the array was converted or not.
        obs_treat : array_like
            Observation of the treatment sample. It is a boolean vector (0 or 1) which indicates weather the sample i-th
            of the array was converted or not.
        alpha : float in interval (0,1)
            Significance level, default 0.05. It is the probability of a type I error, that is wrong rejections if the
            Null Hypothesis is true.

        Returns
        -------
        p_value : float in interval (0,1)
            p-value for the statistical test.
        ci_contr : tuple
            confidence interval for the control group.
        ci_treat : tuple
            confidence interval for the treatment group.
        """

        return self.compare_conv_stats(conv_contr=np.sum(obs_contr), conv_treat=np.sum(obs_treat),
                                       nobs_contr=len(obs_contr), nobs_treat=len(obs_treat))

    def check_homogeneity(self, df, group, cat_cols, verbose=False):
        """
        Check variables homogeneity of the samples considered in the experiment. The goal is to verify homogeneity
        between control and treatment groups. It performs univariate logistic regression per each variable of the input
        samples where the dependent variable is the group variation.

        Parameters
        ----------
        df : pandas DataFrame of shape (n_samples, n_variables)
            Input samples to be checked.
        group : array-like of shape (n_samples,)
            Groups variation of each sample (either 0 or 1).
        cat_cols : list
            List of the column names to be considered as categorical variables.
        verbose : bool
            Print detailed information of the logistic regression.

        Returns
        -------
        stats : pandas DataFrame
            Statistics of the logistic regression (coefficients, p-values, etc.)
        """

        # Select continuous variables
        cont_cols = [c for c in list(df.columns) if c not in cat_cols]

        # Change type to string for categorical variables
        for col in cat_cols:
            df[col] = df[col].astype(str)

        # Adapt categorical variables names for formula
        formula_cat_cols = ["C(" + col + ", Treatment('" + str(df[col].value_counts().idxmax()) + "'))"
                            for col in cat_cols]

        # Select columns names
        formula_cols = cont_cols + formula_cat_cols
        cols = cont_cols + cat_cols

        stats = []

        for col, formula_col in zip(cols, formula_cols):

            # Define formula
            formula = "group ~ " + str(formula_col)

            # Add intercept to the model
            dfcol = add_constant(df[col])

            # Fit the model
            res = logit(formula=formula, data=dfcol).fit(disp=verbose)

            # Convert summary results to html and append them to a list
            res_as_html = res.summary().tables[1].as_html()
            stats.append(pd.read_html(res_as_html, header=0, index_col=0)[0])

        # Concatenate results in a pandas DataFrame
        stats = pd.concat(stats).drop('Intercept')

        return stats

    def bootstrap(self, data, func, rep=500, seed=None):
        """
        Perform bootstrapping on the observed dataset. This technique makes inference about a certain estimate (e.g.
        sample mean) for a certain population parameter (e.g. population mean) by resampling with replacement from the
        observed dataset. This technique does not make assumptions on the observed samples distribution.

        Parameters
        ----------
        data : array_like of shape (n_samples, n_days)
            Input samples for bootstrapping.
        func : function, default np.mean
            Function used to aggregate samples at each bootstrapping iteration. The function must compute its
            aggregation along axis=0.
        rep : int, default 500.
            Number of resampling repetitions.
        seed : int, default None.
            Seed for random state. The function outputs deterministic results if called more times with equal inputs
            while maintaining the same seed.

        Returns
        -------
        stats : pandas DataFrame
            Summary statistics of bootstrapping (median, 2.5 percentile, 97.5 percentile).
        """

        # Reset index to numerical values
        data = pd.DataFrame(data).copy()
        data.reset_index(drop=True)

        # Bootstrap
        boot = [resample(data, replace=True, n_samples=len(data), random_state=(None if seed is None else seed*i))
                for i in np.arange(rep)]

        # Compute statistics of each sample. List shape=(#rep x #days)
        samples_stats = [func(sample) for sample in boot]

        # Rearrange samples_stats in shape: (#days x # rep)
        samples_stats_day = list(zip(*samples_stats))

        # Create table of results
        stats = pd.DataFrame(columns=['median', '2.5 percentile', '97.5 percentile'])

        # Define table of results indexes
        idx = ([''] if data.shape[1] == 1 else boot[0].columns)

        # Fill table of results with median, 2.5 percentile, 97.5 percentile of samples_stats per day
        for i in np.arange(len(samples_stats_day)):
            stats.loc[str(idx[i])] = {'median': np.median(samples_stats_day[i]),
                                      '2.5 percentile': np.percentile(samples_stats_day[i], 2.5),
                                      '97.5 percentile': np.percentile(samples_stats_day[i], 97.5)}

        return stats
