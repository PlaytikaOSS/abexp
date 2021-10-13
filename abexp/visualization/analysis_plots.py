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

from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np


class AnalysisPlot:

    @staticmethod
    def barplot(bars, yerr, figsize=(10, 8), width=0.4, fontsize=14, xlabel=None, ylabel=None, groupslabel=None,
                title=None, rotation=0, capsize=None, legendloc=None):
        """

        Make bars plot with confidence intervals for N groups (A/B/C...) given M segments.

        Parameters
        ----------
        bars : array_like of shape (n_group, n_segments)
            Height of the bars.
        yerr : array_like of shape (n_groups,)
            Lower and upper limit of the confidence interval error bar (y-err_low, y+err_upp) per each group.
        figsize : Tuple, default (10, 8)
            Figure dimension (width, height) in inches.
        width : float, default 0.4
            Width of the bars.
        fontsize : float, default None
            Font size of the elements in the figure.
        xlabel : list of length n_segments, default None
            List of labels for the segments on the x axis.
        ylabel : string, default None
            Label for the y axis.
        groupslabel : list of length n_groups, default None
            List of labels for group variations.
        title : str, default None
            Title of the figure.
        rotation : float, default 0
            Degree of rotation for xlabels.
        capsize : float, default None
            Width of the confidence intervals cap.
        legendloc : str, default None
            Location of the legend. Possible values: 'center', 'best', 'upper left', 'upper right', 'lower left',
            'lower right', 'upper center', 'lower center', 'center left', 'center right'.

        Returns
        -------
        fig :  matplotlib figure
            Output figure
        """

        # Generate labels if arguments are not specified
        if xlabel is None:
            xlabel = ['segment ' + str(i + 1) for i in np.arange(len(bars[0]))]
        if groupslabel is None:
            groupslabel = ['group ' + s for s in list(ascii_uppercase)]
        if capsize is None:
            capsize = width * 100 / len(bars[0])

        # Create figure and grid
        fig, ax = plt.subplots(figsize=figsize)
        ax.yaxis.grid()
        ax.set_axisbelow(True)

        # Define x position of bars
        x = np.arange(len(bars[0]))
        offsets = np.arange(start=-len(bars) + 1, stop=len(bars), step=2)
        xpos = [x + offset / len(bars) * width for offset in offsets]

        for i in np.arange(len(bars)):

            # Calculate the positive/negative errors from the bars heights: err_upp = y_upp - y, err_low = y - y_low.
            # This is computed because plt.bar takes as input the height y and the positive/negative errors.
            for j in np.arange(len(bars[0])):
                yerr[i][0][j] = bars[i][j] - yerr[i][0][j]
                yerr[i][1][j] = yerr[i][1][j] - bars[i][j]

            # Plot groups bars
            plt.bar(xpos[i], bars[i], width=2 / len(bars) * width, yerr=yerr[i], capsize=capsize, label=groupslabel[i])

        # Add title and legend
        if title:
            plt.title(title, fontsize=fontsize)
        plt.legend(fontsize=fontsize, loc=legendloc)

        # Add axis labels and custom axis tick labels
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks([r for r in range(len(bars[0]))], xlabel, fontsize=fontsize, rotation=rotation)

        return fig

    @staticmethod
    def forest_plot(y, ci, figsize=(10, 8), fontsize=14, xlabel=None, ylabel=None, annotation=None,
                    annotationlabel=None, title=None, rotation=0, capsize=None, legendloc=None, marker='s'):
        """

        Make forest plot with confidence intervals for N groups.

        Parameters
        ----------
        y : array_like of shape (n_groups,)
            Vertical coordinate of the central data points
        ci : array_like of shape (n_groups,)
            Confidence intervals +/- values for y.
        figsize : Tuple, default (10, 8)
            Figure dimension (width, height) in inches.
        fontsize : float, default None
            Font size of the elements in the figure.
        xlabel : list of length n_groups, default None
            List of labels for the groups on the x axis.
        ylabel : string, default None
            Label for the y axis.
        annotation : list of length n_groups
            Annotation value to be displayed per each bar
        annotationlabel : list of shape
            Annotation label description to be displayed per each bar
        title : str, default None
            Title of the figure.
        rotation : float, default 0
            Degree of rotation for xlabels.
        capsize : float, default None
            Width of the confidence intervals cap.
        legendloc : str, default None
            Location of the legend. Possible values: 'center', 'best', 'upper left', 'upper right', 'lower left',
            'lower right', 'upper center', 'lower center', 'center left', 'center right'.
        marker : str, default 's'
            Marker style. Possible values: '.', ',', 'o', '8', 's', 'P', 'h', 'H', '+', 'x', 'X', 'd', 'D', '_', etc.

        Returns
        -------
        fig :  matplotlib figure
            Output figure
        """

        # Generate default arguments if not specified
        if xlabel is None:
            xlabel = ['group ' + str(i + 1) for i in np.arange(len(y))]
        if capsize is None:
            capsize = 10
        annotationlabel = '' if annotationlabel is None else annotationlabel + '\n'

        # Create figure and grid
        fig, ax = plt.subplots(figsize=figsize)
        ax.yaxis.grid()
        ax.set_axisbelow(True)

        # Define x position of bars
        x = np.arange(len(y))

        # Plot groups bars
        for i in np.arange(len(y)):
            plt.errorbar(x=x[i], y=y[i], yerr=ci[i], capsize=capsize, linestyle='None', marker=marker, ms=10, mew=2,
                         label=xlabel[i])

        # Annotate additional statistics to forest plot
        if annotation is not None:
            for xpos, ypos, ann in zip(x, y, annotation):
                plt.annotate(annotationlabel + str(ann), (xpos, ypos), xytext=(10, 0), textcoords="offset points",
                             ha='left', va='center', fontsize=fontsize)

        # Add title and legend
        if title:
            plt.title(title, fontsize=fontsize)
        plt.legend(loc=legendloc, fontsize=fontsize)

        # Add axis labels and custom axis tick labels
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks([r for r in range(len(y))], xlabel, fontsize=fontsize, rotation=rotation)
        plt.xlim(-1, len(y))

        return fig

    @staticmethod
    def timeseries_plot(y, ci, figsize=(15, 10), fontsize=14, xlabel=None, ylabel=None, groupslabel=None, title=None,
                        rotation=45, capsize=None, legendloc=None):
        """

        Make time series plot with confidence intervals for N groups.

        Parameters
        ----------
        y : array_like of shape (n_groups, n_day)
            Input time series
        ci : array_like of shape (n_group, n_days)
            Confidence intervals +/- values for y.
        figsize : Tuple, default (10, 8)
            Figure dimension (width, height) in inches.
        fontsize : float, default None
            Font size of the elements in the figure.
        xlabel : list of length n_days, default None
            List of labels for the days on the x axis.
        ylabel : string, default None
            Label for the y axis.
        groupslabel : list of length n_groups, default None
            List of labels for group variations.
        title : str, default None
            Title of the figure.
        rotation : float, default 45
            Degree of rotation for xlabels.
        capsize : float, default None
            Width of the confidence intervals cap.
        legendloc : str, default None
            Location of the legend. Possible values: 'center', 'best', 'upper left', 'upper right', 'lower left',
            'lower right', 'upper center', 'lower center', 'center left', 'center right'.

        Returns
        -------
        fig :  matplotlib figure
            Output figure
        """

        # Generate labels if argument is not specified
        if xlabel is None:
            xlabel = ['day' + str(i + 1) for i in np.arange(len(y[0]))]
        if groupslabel is None:
            groupslabel = ['group ' + s for s in list(ascii_uppercase)]
        if capsize is None:
            capsize = 5

        # Create figure and grid
        fig, ax = plt.subplots(figsize=figsize)
        ax.yaxis.grid()
        ax.set_axisbelow(True)

        # Define x position of bars
        x = np.arange(len(y[0]))

        # Plot groups bars
        for i in np.arange(len(y)):
            plt.errorbar(x=x, y=y[i], yerr=ci[i], capsize=capsize, marker=".", ms=10, mew=2, label=groupslabel[i])

        # Add title and legend
        if title:
            plt.title(title, fontsize=fontsize)
        plt.legend(loc=legendloc, fontsize=fontsize)

        # Add axis labels and custom axis tick labels
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks([r for r in range(len(y[0]))], xlabel, fontsize=fontsize, rotation=rotation)
        plt.xlim(-1, len(y[0]))
        plt.legend(fontsize=fontsize)

        return fig
