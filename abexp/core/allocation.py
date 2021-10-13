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
from stochatreat import stochatreat


class Allocator:
    """
    This class provides some utils to be used before running A/B test experiments. Groups allocation is the
    process that assigns (allocates) a list of users either to group A (e.g. control) or to group B (e.g. treatment).
    This class provides functionalities to randomly allocate users in two or more groups (A/B/C/...).
    """
    def __init__(self):
        pass

    @staticmethod
    def complete_randomization(user_id, ngroups=2, prop=None, seed=None):
        """
        Random allocate users in n groups.

        Parameters
        ----------
        user_id : array_like
            Array of user ids.
        ngroups : int
            Number of group variations, default 2.
        prop : array_like of floats in interval (0,1)
            Proportions of users in each group. By default, each group has the same amount of users. Proportion should
            sum up to 1.
        seed : int, default None.
            Seed for random state. The function outputs deterministic results if called more times with equal inputs
            while maintaining the same seed.

        Returns
        -------
        df : pd.DataFrame
            Dataset of user ids with additional column for the group variation
        stats : pd.DataFrame
            Statistics of the number of users contained in each group
        """

        # Generate equal proportions if argument is not specified
        if prop is None:
            prop = [1 / ngroups] * ngroups

        assert (ngroups == len(prop)), 'The length of the vector of proportions should be equal to the number of groups'
        np.testing.assert_almost_equal(sum(prop), 1, err_msg='Proportions do not sum to 1')

        # Generate groups
        groups = []
        for i in np.arange(ngroups):
            groups.append([i] * round(len(user_id) * prop[i]))

        # Flatten groups in a 1D array
        groups = np.array([item for subgroups in groups for item in subgroups])

        # Define random seed
        np.random.seed(seed)

        # Shuffle groups
        np.random.shuffle(groups)

        # The length of groups must be equal to the length of user_id. If not cut/add elements to groups.
        if len(groups) > len(user_id):
            groups = groups[:len(user_id)]
            print('cutting')
        leftover = np.arange(len(user_id) - len(groups))
        groups = np.append(groups, leftover)

        assert (len(groups) == len(user_id)), 'Randomization cannot be achieved. Mismatch between length of user_if ' \
                                              'and allocated groups'''

        # Compute number of users per group
        unique, counts = np.unique(groups, return_counts=True)

        # Define DataFrames to return
        df = pd.DataFrame({'user_id': user_id, 'group': groups})
        stats = pd.DataFrame({'group': unique, '#users': counts}).set_index(['group']).T

        return df, stats

    @staticmethod
    def blocks_randomization(df, id_col, stratum_cols, ngroups=2, prop=[None], seed=None):
        """
        Random allocate users within a block in n groups. Users with similar characteristics (features) define a block,
        and randomization is conducted within a block. This enables balanced and homogeneous groups of similar sizes.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset of users.
        id_col : str
            Column name of the user ids.
        stratum_cols : list
            List of column names to be stratified over
        ngroups : int
            Number of group variations, default 2.
        prop : array_like of floats in interval (0,1)
            Proportions of users in each group. By default, each group has the same amount of users.
        seed : int, default None.
            Seed for random state. The function outputs deterministic results if called more times with equal inputs
            while maintaining the same seed.

        Returns
        -------
        df : pd.DataFrame
            Dataset of users with additional column for the group variation
        stats : pd.DataFrame
            Statistics of the number of users contained in each group
        """

        # Asserts on column names
        assert('group' not in df.columns), "You cannot have 'group' as column name."
        assert('treat' not in df.columns), "You cannot have 'treat' as column name."

        df = pd.DataFrame(df).copy()

        # Randomly assign groups by neighborhoods and dummy status.
        treats = stochatreat(data=df, idx_col=id_col, stratum_cols=stratum_cols, treats=ngroups, probs=prop,
                             random_state=seed,  misfit_strategy='stratum')

        # Merge back with original data and drop the stratum id columns
        df = df.merge(treats, how='left', on=id_col)
        df.drop(columns=['stratum_id'], inplace=True)
        df.rename(columns={"treat": "group"}, inplace=True)

        # Computer
        stats = df.groupby(stratum_cols)['group'].value_counts().unstack()

        return df, stats
