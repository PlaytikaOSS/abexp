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

from abexp.core.allocation import Allocator


def test_random_allocation():
    """ Test complete randomization. """

    df, stats = Allocator.complete_randomization(np.arange(12), 5, seed=42)

    assert(all(df['group'] == [4, 0, 2, 0, 3, 1, 4, 2, 1, 3, 0, 1])), \
        'Error: computed df = {}'.format(df['group'])
    assert(all(stats.columns.values == [0, 1, 2, 3, 4])), \
        'Error: computed stats group = {}'.format(stats.columns.values)
    assert(all(stats.loc['#users'].values == [3, 3, 2, 2, 2])), \
        'Error: computed stats #users = {}'.format(stats.loc['#users'].values)

    df, stats = Allocator.complete_randomization(np.arange(10), 2, seed=42)

    assert (all(df['group'] == [1, 0, 1, 0, 1, 0, 1, 0, 0, 1])), \
        'Error: computed df: \n{}'.format(df['group'])
    assert (all(stats.columns.values == [0, 1])), \
        'Error: computed stats group: \n{}'.format(stats.columns.values)
    assert (all(stats.loc['#users'].values == [5, 5])), \
        'Error: computed stats #users: \n{}'.format(stats.loc['#users'].values)

    df, stats = Allocator.complete_randomization(np.arange(20), 6, prop=[0.13, 0.12, 0.21, 0.19, 0.3, 0.05], seed=42)

    assert (all(df['group'] == [0, 4, 4, 0, 2, 2, 3, 1, 4, 4, 4, 0, 3, 5, 1, 3, 2, 3, 4, 2])), \
        'Error: computed df = {}'.format(df['group'])
    assert (all(stats.columns.values == [0, 1, 2, 3, 4, 5])), \
        'Error: computed stats group = {}'.format(stats.columns.values)
    assert (all(stats.loc['#users'].values == [3, 2, 4, 4, 6, 1])), \
        'Error: computed stats #users = {}'.format(stats.loc['#users'].values)


def test_blocks_allocation():
    """ Test block randomization. """
    np.random.seed(42)
    df = pd.DataFrame(data={'user_id': list(range(30)),
                            'experience': np.random.randint(100, 500, size=30),
                            'level': np.random.randint(1, 4, size=30)})

    df, stats = Allocator.blocks_randomization(df, id_col='user_id', stratum_cols='level', seed=30)

    assert (all(df['group'] == [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0,
                                1])), 'Error: computed df: \n{}'.format(df['group'])
    assert(all(stats[0] == [5, 5, 5])), 'Error: computed stats group 0: \n{}'.format(stats[0])
    assert(all(stats[1] == [4, 5, 6])), 'Error: computed stats group 1: \n{}'.format(stats[1])
    assert(all(stats.index == [1, 2, 3])), 'Error: computed stats index: \n{}'.format(stats.index)

    np.random.seed(42)
    df = pd.DataFrame(data={'user_id': list(range(100)),
                            'experience': np.random.randint(100, 500, size=100),
                            'is_paying': np.random.randint(0, 2, size=100),
                            'level': np.random.randint(1, 4, size=100)})

    df, stats = Allocator.blocks_randomization(df, id_col='user_id', stratum_cols=['level', 'is_paying'], ngroups=5,
                                               seed=42, prop=[0.2, 0.2, 0.2, 0.2, 0.2])

    assert (all(df['group'][:30] == [4, 2, 2, 1, 3, 4, 1, 3, 2, 4, 3, 0, 3, 2, 4, 3, 0, 0, 2, 3, 2, 1, 1, 0, 3, 3, 4, 3,
                                     0, 0])), 'Error: computed df: \n{}'.format(df['group'])
    assert (all(stats[0] == [4, 3, 4, 1, 4, 3])), 'Error: computed stats group 0: \n{}'.format(stats[0])
    assert (all(stats[1] == [3, 3, 4, 3, 5, 3])), 'Error: computed stats group 1: \n{}'.format(stats[1])
    assert (all(stats[2] == [4, 3, 4, 3, 5, 2])), 'Error: computed stats group 2: \n{}'.format(stats[2])
    assert (all(stats[3] == [3, 3, 4, 3, 5, 2])), 'Error: computed stats group 3: \n{}'.format(stats[3])
    assert (all(stats[4] == [4, 3, 4, 1, 5, 2])), 'Error: computed stats group 4: \n{}'.format(stats[4])
