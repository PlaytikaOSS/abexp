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

from math import lgamma

import numpy as np
import pandas as pd
import pymc3 as pm
from arviz import hdi
from scipy.stats import beta


class BayesianAnalyzer:
    """
    This class provides tools to perform analysis after A/B test experiments with bayesian statistical approach. It
    handles both the case of means comparison and conversions comparison with closed-form-solutions or simulation.
    Bayesian analysis does not make any normality assumptions on the sample data.
    """
    def __init__(self):
        pass

    # code snippet below is taken from https://gist.github.com/arsatiki/1395348/f0275f529d322d3c23e18201f26890f5a09dcb51
    # start code snippet (credits to Antti Rasinen)
    def _h(self, a, b, c, d):
        num = lgamma(a + c) + lgamma(b + d) + lgamma(a + b) + lgamma(c + d)
        den = lgamma(a) + lgamma(b) + lgamma(c) + lgamma(d) + lgamma(a + b + c + d)
        return np.exp(num - den)

    def _g0(self, a, b, c):
        return np.exp(lgamma(a + b) + lgamma(a + c) - (lgamma(a + b + c) + lgamma(a)))

    def _hiter(self, a, b, c, d):
        while d > 1:
            d -= 1
            yield self._h(a, b, c, d) / d

    def _g(self, a, b, c, d):
        return self._g0(a, b, c) + sum(self._hiter(a, b, c, d))
    # end code snippet (credits to Antti Rasinen)

    def _calc_prob(self, beta1, beta2):
        return self._g(beta1.args[0], beta1.args[1], beta2.args[0], beta2.args[1])

    def compare_conv(self, conv_contr, conv_treat, nobs_contr, nobs_treat):
        """
        Compare conversions from statistics. Compare the conversions of the control group versus the conversions of the
        treatment group. The result is computed via bayesian analysis with a closed-form solution based on the concept
        of conjugate priors.

        Reference paper: John Cook, Exact calculation of beta inequalities (2005).

        Parameters
        ----------
        conv_contr : int > 0
            Number of conversions in the control group.
        conv_treat : int > 0
            Number of conversions in the treatment group.
        nobs_contr : int > 0
            Number of observations in the control group
        nobs_treat : int > 0
            Number of observations in the treatment group

        Returns
        -------
        prob : float in interval (0,1)
            probability that treatment group is better than control group
        lift : float in interval (0,1)
            lift between the two groups
        """

        # Define prior Beta distributions for the two groups with params: a=convs+1, b=nsample-convs+1
        a_contr = conv_contr + 1
        a_treat = conv_treat + 1

        b_contr = nobs_contr - conv_contr + 1
        b_treat = nobs_treat - conv_treat + 1

        beta_contr = beta(a_contr, b_contr)
        beta_treat = beta(a_treat, b_treat)

        # Compute probability for test to be better than control
        prob = self._calc_prob(beta_treat, beta_contr)

        # Compute lift between treatment and control groups
        lift = (beta_treat.mean() - beta_contr.mean()) / beta_contr.mean()

        return prob, lift

    def compare_mean(self, obs_contr, obs_treat, n=50000):
        """
        Compare means from observed samples. Compare the mean of the control group versus the mean of the treatment
        group. The result is computed via bayesian analysis with Markov chain Monte Carlo (MCMC) simulation.

        Reference paper: John K. Kruschke, Bayesian Estimation Supersedes the t Test (2012)

        Parameters
        ----------
        obs_contr : array_like
            Observation of first sample
        obs_treat : array_like
            Observation of second sample
        n: int, default 500000
            The number of samples to draw in MCMC

        Returns
        -------
        prob : float in interval (0,1)
            Probability that treatment group is better than control group
        lift : float in interval (0,1)
            Lift between the two groups
        diff_means : float
            Difference of means. The treatment group mean - control group mean.
        ci : Tuple of floats
            Credible intervals. Lower and upper values of the interval [low, high].
        """

        # Concatenate samples
        y = np.concatenate((obs_contr, obs_treat))

        with pm.Model() as model:  # noqa: F841

            # Priors distributions for the mean parameter
            mu_m = np.mean(y)
            mu_p = 0.000001 * 1 / np.std(y) ** 2
            mean_contr = pm.Normal('mean_contr', mu=mu_m, sd=mu_p)
            mean_treat = pm.Normal('mean_treat', mu=mu_m, sd=mu_p)

            # Prior distribution for the standard deviation parameter
            sigma_low = np.std(y) / 1000
            sigma_high = np.std(y) * 1000
            std_contr = pm.Uniform('std_contr', lower=sigma_low, upper=sigma_high)
            std_treat = pm.Uniform('std_treat', lower=sigma_low, upper=sigma_high)

            # Prior distribution for the nu parameter
            nu = pm.Exponential('nu_minus_one', 1 / 29.) + 1

            # Prior distributions as student-t
            group_contr = pm.StudentT('group_contr', nu=nu, mu=mean_contr, lam=std_contr**-2,  # noqa: F841
                                      observed=obs_contr)
            group_treat = pm.StudentT('group_treat', nu=nu, mu=mean_treat, lam=std_treat**-2,  # noqa: F841
                                      observed=obs_treat)

            # Define difference of means
            delta = pm.Deterministic('delta', mean_treat - mean_contr)  # noqa: F841

            # Run Markov Chain Monte Carlo simulations
            trace = pm.sample(draws=n, tune=1000, init='adapt_diag', progressbar=False)

        # Extract samples from the simulation
        delta_samples = trace['delta']

        # Compute probability that control mean is grater than treatment mean
        prob = np.mean(delta_samples > 0)

        # Compute lift between treatment and control groups
        lift = (np.mean(obs_treat) - np.mean(obs_contr)) / np.mean(obs_contr)

        # Compute difference of means
        diff_means = np.mean(obs_treat) - np.mean(obs_contr)

        # Compute credible intervals
        ci = hdi(delta_samples, hdi_prob=.95)

        return prob, lift, diff_means, ci


class BayesianGLMAnalyzer:
    """
    The class provides tools to perform analysis after A/B test experiments with bayesian statistical approach. It
    provides techniques based on bayesian generalized linear model (GLM) with multivariate and hierarchical regression.
    Bayesian analysis does not make any normality assumptions on the sample data.
    """
    def __init__(self):
        pass

    def multivariate_regression(self, df, kpi_col, family=pm.glm.families.StudentT()):
        """
        Compare means from observed samples. Compare the mean of the control group versus the mean of the treatment
        group. The result is computed via bayesian generalized linear model (GLM) with robust multivariate regression.

        Parameters
        ----------
        df : pandas DataFrame of shape (n_samples, n_variables)
            Input samples data set
        kpi_col : str
            Column name in the input dataset of the kpi
        family : pymc3.glm.families, default StudentT
            Priors family distribution

        Returns
        -------
        stats : pandas DataFrame
            Summary statistics of the model
        """

        # Get columns names for regression coefficients
        reg_columns = df.columns.drop(kpi_col)

        # Get kpi
        y = df[kpi_col].values  # noqa: F841

        with pm.Model() as model:  # noqa: F841

            # Define priors regression coefficients
            priors = {col: pm.Normal.dist(mu=df[col].mean(), sigma=df[col].std()) for col in reg_columns}

            # Define model
            model = pm.glm.GLM.from_formula('y ~ ' + " + ".join(reg_columns), df, priors=priors,  # noqa: F841
                                            family=family)
            trace = pm.sample(draws=2000, tune=1000, init='adapt_diag', progressbar=False)

        # Table with statistics
        trace_df = pm.backends.tracetab.trace_to_dataframe(trace)
        stats = pd.DataFrame(trace_df.describe().drop('count').T)
        stats['Prob<0'] = (trace_df < 0).mean()
        stats['Prob>0'] = (trace_df > 0).mean()

        return stats

    def hierarchical_regression(self, df, group_col, cat_col, kpi_col):
        """
        Compare means from observed samples. Compare the mean of the control group versus the mean of the treatment
        group. The result is computed via bayesian hierarchical generalized linear model (GLM).

        Parameters
        ----------
        df : pandas DataFrame of shape (n_samples, n_variables)
            Input samples data set
        group_col : str
            Column name in the input dataset of the group variation
        cat_col : str
            Column name in the input dataset of the categorical variable
        kpi_col : str
            Column name in the input dataset of the kpi

        Returns
        -------
        stats : pandas DataFrame
            Summary statistics of the model
        """
        # Set cat_col type as categorical
        df[cat_col] = df[cat_col].astype('category')

        # Generate indexes for the categorical column
        cat_col_idx = df[cat_col].cat.codes

        # Get unique values of the categorical column
        cat_col_unique = df[cat_col].unique()

        with pm.Model() as model:  # noqa: F841

            # Hyperpriors
            mu_a = pm.Normal('mu_alpha', mu=0, sigma=1)
            mu_b = pm.Normal('mu_beta', mu=0, sigma=1)
            sigma_a = pm.HalfCauchy('sigma_alpha', beta=1)
            sigma_b = pm.HalfCauchy('sigma_beta', beta=1)

            a = pm.Normal('alpha', mu=mu_a, sigma=sigma_a, shape=len(cat_col_unique))
            b = pm.Normal('beta', mu=mu_b, sigma=sigma_b, shape=len(cat_col_unique))

            # Model error
            eps = pm.HalfCauchy('eps', beta=1)

            # Expected value
            y = a[cat_col_idx] + b[cat_col_idx] * df[group_col].values

            # Data likelihood
            y_like = pm.Normal('y_like', mu=y, sigma=eps, observed=df[kpi_col])  # noqa: F841

            # Run MCMC
            hierarchical_trace = pm.sample(draws=2000, tune=1000, init='adapt_diag', progressbar=False)

            # Table with statistics
            trace_df = pm.backends.tracetab.trace_to_dataframe(hierarchical_trace)
            stats = pd.DataFrame(trace_df.describe().drop('count').T)
            stats['Prob<0'] = (trace_df < 0).mean()
            stats['Prob>0'] = (trace_df > 0).mean()
            for i in cat_col_idx.unique():
                stats.rename(index={'alpha__' + str(i): 'alpha__' + df[cat_col].unique()[i]}, inplace=True)
                stats.rename(index={'beta__' + str(i): 'beta__' + df[cat_col].unique()[i]}, inplace=True)

            return stats
