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

import matplotlib.pyplot as plt
import numpy as np

from abexp.visualization.analysis_plots import AnalysisPlot


def test_barplot():
    """ Test bar plot. """
    labels = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']
    groups = ['group A', 'group B', 'group C', 'group D']

    # Choose the height of the  bars
    bars1 = [200, 340, 300, 360, 340, 800]
    bars2 = [250, 320, 340, 270, 400, 100]
    bars3 = [250, 320, 340, 270, 400, 100]
    bars4 = [250, 320, 340, 270, 600, 700]

    # Choose the height of the error bars
    yerr1 = [[200-5,  340-5,  300-5,  360-8,  340-12,  800-50],
             [200+20, 340+20, 300+20, 360+70, 340+100, 800+50]]
    yerr2 = [[250-50, 320-50, 340-50, 270-34, 400-90,  100-3],
             [250+10, 320+10, 340+10, 270+30, 400+60,  100+20]]
    yerr3 = [[250-50, 320-50, 340-50, 270-34, 400-90,  100-3],
             [250+10, 320+10, 340+10, 270+30, 400+60,  100+20]]
    yerr4 = [[250-5,  320-5,  340-5,  270-8,  600-12,  700-50],
             [250+20, 320+20, 340+20, 270+70, 600+100, 700+50]]

    bars = [bars1, bars2, bars3, bars4]
    yerr = [yerr1, yerr2, yerr3, yerr4]

    fig1 = AnalysisPlot.barplot(bars, yerr)

    assert (type(fig1) == plt.Figure)

    fig2 = AnalysisPlot.barplot(bars, yerr, figsize=(10, 8), width=0.4, fontsize=14, xlabel=labels, ylabel='heights',
                                groupslabel=groups, title='barplot', rotation=45, capsize=10, legendloc='center')

    assert (type(fig2) == plt.Figure)


def test_forest_plot():
    """ Test forest plot. """

    labels = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']
    y = np.random.choice(np.arange(1, 7, 0.1), 6)
    ci = np.random.choice(np.arange(0.5, 3, 0.1), 6)
    p_val = np.around(np.random.choice(np.arange(0.5, 1, 0.1), 6), 4)

    fig1 = AnalysisPlot.forest_plot(y, ci)

    assert (type(fig1) == plt.Figure)

    fig2 = AnalysisPlot.forest_plot(y, ci, figsize=(18, 12), fontsize=14, xlabel=labels, ylabel='diff arpu',
                                    annotation=p_val, annotationlabel='p-value', title='Forest Plot', rotation=45,
                                    capsize=10, legendloc='center', marker='.')

    assert (type(fig2) == plt.Figure)


def test_timeseries_plot():
    """ Test timeseries plot. """
    labels = [str(i) for i in np.arange(30)]

    y1 = np.random.choice(np.arange(1, 7, 0.1), 30)
    ci1 = np.random.choice(np.arange(0.5, 1, 0.1), 30)

    y2 = np.random.choice(np.arange(1, 7, 0.1), 30) + 10
    ci2 = np.random.choice(np.arange(0.5, 1, 0.1), 30)

    y = [y1, y2]
    ci = [ci1, ci2]

    fig1 = AnalysisPlot.timeseries_plot(y, ci)

    assert (type(fig1) == plt.Figure)

    fig2 = AnalysisPlot.timeseries_plot(y, ci, figsize=(15, 10), fontsize=14, xlabel=labels, ylabel='y_val',
                                        groupslabel=labels, title='Time Series Plot', rotation=0, capsize=10,
                                        legendloc='center')

    assert (type(fig2) == plt.Figure)
