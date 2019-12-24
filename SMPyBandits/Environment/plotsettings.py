# -*- coding: utf-8 -*-
""" plotsettings: use it like this, in the Environment folder:

>>> import sys; sys.path.insert(0, '..')
>>> from .plotsettings import BBOX_INCHES, signature, maximizeWindow, palette, makemarkers, add_percent_formatter, wraptext, wraplatex, legend, show_and_save, nrows_ncols
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from textwrap import wrap
import os.path

import matplotlib as mpl
# mpl.use('Agg')  # XXX is it a good idea? Nope, use "export MPLBACKEND='Agg'" in your bashrc ... Cf. http://stackoverflow.com/a/4935945/ and http://matplotlib.org/faq/usage_faq.html#what-is-a-backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import numpy as np
import seaborn as sns

# Customize here if you want a signature on the titles or xlabel, of each plot
from datetime import datetime
import locale  # See this bug, http://numba.pydata.org/numba-doc/dev/user/faq.html#llvm-locale-bug
locale.setlocale(locale.LC_TIME, 'C')
monthyear = "{:%b.%Y}".format(datetime.today()).title()  #: Month.Year date

from os import getenv

# Backup figure objects
from pickle import dump as pickle_dump

if getenv('DEBUG', 'False') == 'True':
    signature = "\n(By Lilian Besson, {}, cf. SMPyBandits.GitHub.io - MIT Licensed)".format(monthyear)  #: A small string to use as a signature
else:
    signature = ""

DPI = 120  #: DPI to use for the figures
# FIGSIZE = (19.80, 10.80)  #: Figure size, in inches!
FIGSIZE = (16, 9)  #: Figure size, in inches!
# FIGSIZE = (12.4, 7)  #: Figure size, in inches!
# FIGSIZE = (8, 6)  #: Figure size, in inches!
# FIGSIZE = (8, 4.5)  #: Figure size, in inches!

# Customize the colormap
HLS = True  #: Use the HLS mapping, or HUSL mapping
VIRIDIS = False  #: Use the Viridis colormap

# Bbox in inches. Only the given portion of the figure is saved. If 'tight', try to figure out the tight bbox of the figure.
BBOX_INCHES = "tight"  #: Use this parameter for bbox
BBOX_INCHES = None

if __name__ != '__main__':
    # use a clever color palette, eg http://seaborn.pydata.org/api.html#color-palettes
    sns.set(context="talk", style="whitegrid", palette="hls" if HLS else "husl", font="sans-serif", font_scale=1.05)

    # Use tex by default http://matplotlib.org/2.0.0/users/dflt_style_changes.html#math-text
    # mpl.rcParams['text.usetex'] = True  # XXX force use of LaTeX
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams['font.sans-serif'] = "DejaVu Sans"
    mpl.rcParams['mathtext.fontset'] = "cm"
    mpl.rcParams['mathtext.rm'] = "serif"

    # Configure size for axes and x and y labels
    # Cf. https://stackoverflow.com/a/12444777/
    # mpl.rcParams['axes.labelsize']  = "small"
    # mpl.rcParams['xtick.labelsize'] = "x-small"
    # mpl.rcParams['ytick.labelsize'] = "x-small"
    # mpl.rcParams['figure.titlesize'] = "small"

    mpl.rcParams['axes.labelsize']  = "medium"
    mpl.rcParams['lines.linewidth']  = 10
    mpl.rcParams['lines.markersize']  = 14
    mpl.rcParams['xtick.labelsize'] = "large"
    mpl.rcParams['ytick.labelsize'] = "large"
    mpl.rcParams['figure.titlesize'] = "large"

    # Configure the DPI of all images, once and for all!
    mpl.rcParams['figure.dpi'] = DPI
    # print(" - Setting dpi of all figures to", DPI, "...")  # DEBUG

    # Configure figure size, even of if saved directly and not displayed, use HD screen
    # cf. https://en.wikipedia.org/wiki/Computer_display_standard
    mpl.rcParams['figure.figsize'] = FIGSIZE
    # print(" - Setting 'figsize' of all figures to", FIGSIZE, "...")  # DEBUG

    # XXX Set up a discrete version of the Viridis map for axes.prop_cycle


def palette(nb, hls=HLS, viridis=VIRIDIS):
    """ Use a smart palette from seaborn, for nb different plots on the same figure.

    - Ref: http://seaborn.pydata.org/generated/seaborn.hls_palette.html#seaborn.hls_palette

    >>> palette(10, hls=True)  # doctest: +ELLIPSIS
    [(0.86..., 0.37..., 0.33...), (0.86...,.65..., 0.33...), (0.78..., 0.86...,.33...), (0.49..., 0.86...,.33...), (0.33..., 0.86...,.46...), (0.33..., 0.86...,.74...), (0.33..., 0.68..., 0.86...) (0.33..., 0.40..., 0.86...) (0.56..., 0.33..., 0.86...) (0.84..., 0.33..., 0.86...)]
    >>> palette(10, hls=False)  # doctest: +ELLIPSIS
    [[0.96..., 0.44..., 0.53...], [0.88..., 0.52..., 0.19...], [0.71..., 0.60..., 0.19...], [0.54..., 0.65..., 0.19...], [0.19..., 0.69..., 0.34...], [0.20..., 0.68..., 0.58...],[0.21..., 0.67..., 0.69...], [0.22..., 0.65..., 0.84...], [0.55..., 0.57..., 0.95...], [0.85..., 0.44..., 0.95...]]
    >>> palette(10, viridis=True)  # doctest: +ELLIPSIS
    [(0.28..., 0.13..., 0.44...), (0.26..., 0.24..., 0.52...), (0.22..., 0.34..., 0.54...), (0.17..., 0.43..., 0.55...), (0.14..., 0.52..., 0.55...), (0.11..., 0.60..., 0.54...), (0.16..., 0.69..., 0.49...), (0.31..., 0.77..., 0.41...), (0.52..., 0.83..., 0.28...), (0.76..., 0.87..., 0.13...)]

    - To visualize:

    >>> sns.palplot(palette(10, hls=True))  # doctest: +SKIP
    >>> sns.palplot(palette(10, hls=False))  # use HUSL by default  # doctest: +SKIP
    >>> sns.palplot(palette(10, viridis=True))  # doctest: +SKIP
    """
    if viridis:
        return sns.color_palette('viridis', nb)
    else:
        return sns.hls_palette(nb + 1)[:nb] if hls else sns.husl_palette(nb + 1)[:nb]


def makemarkers(nb):
    """ Give a list of cycling markers. See http://matplotlib.org/api/markers_api.html

    .. note:: This what I consider the *optimal* sequence of markers, they are clearly differentiable one from another and all are pretty.

    Examples:

    >>> makemarkers(7)
    ['o', 'D', 'v', 'p', '<', 's', '^']
    >>> makemarkers(12)
    ['o', 'D', 'v', 'p', '<', 's', '^', '*', 'h', '>', 'o', 'D']
    """
    allmarkers = ['o', 'D', 'v', 'p', '<', 's', '^', '*', 'h', '>']
    longlist = allmarkers * (1 + int(nb / float(len(allmarkers))))  # Cycle the good number of time
    return longlist[:nb]  # Truncate


#: Default parameter for legend(): if True, the legend is placed at the right side of the figure, not on it.
#: This is almost mandatory for plots with more than 10 algorithms (good for experimenting, bad for publications).
PUTATRIGHT = True
PUTATRIGHT = False

#: Shrink factor if the legend is displayed on the right of the plot.
#:
#: .. warning:: I still don't really understand how this works. Just manually decrease if the legend takes more space (i.e., more algorithms with longer names)
SHRINKFACTOR = 0.60
SHRINKFACTOR = 0.65
SHRINKFACTOR = 0.70
SHRINKFACTOR = 0.75

#: Default parameter for maximum number of label to display in the legend INSIDE the figure
MAXNBOFLABELINFIGURE = 8


def legend(putatright=PUTATRIGHT, fontsize="large",
        shrinkfactor=SHRINKFACTOR, maxnboflabelinfigure=MAXNBOFLABELINFIGURE,
        fig=None, title=None
    ):
    """plt.legend() with good options, cf. http://matplotlib.org/users/recipes.html#transparent-fancy-legends.

    - It can place the legend to the right also, see https://stackoverflow.com/a/4701285/.
    """
    try:
        len_leg = len(plt.gca().get_legend_handles_labels()[1])
        putatright = len_leg > maxnboflabelinfigure
        if len_leg > maxnboflabelinfigure: print("Warning: forcing to use putatright = {} because there is {} items in the legend.".format(putatright, len_leg))  # DEBUG
    except (ValueError, AttributeError, IndexError) as e:
        # print("    e =", e)  # DEBUG
        pass
    if fig is None:
        # fig = plt.gcf()
        fig = plt  # HACK
    if putatright:
        try:
            # Shrink current axis by 20% on xaxis and 10% on yaxis
            delta_rect = (1. - shrinkfactor)/6.25
            # XXX rect = [left, bottom, right, top] in normalized (0, 1) figure coordinates.
            fig.tight_layout(rect=[delta_rect, delta_rect, shrinkfactor, 1 - 2*delta_rect])
            # Put a legend to the right of the current axis
            fig.legend(loc='center left', numpoints=1, fancybox=True, framealpha=0.8, bbox_to_anchor=(1, 0.5), title=title, fontsize=fontsize)
        except:
            fig.legend(loc='best', numpoints=1, fancybox=True, framealpha=0.8, title=title, fontsize=fontsize)
    else:
        fig.legend(loc='best', numpoints=1, fancybox=True, framealpha=0.8, title=title, fontsize=fontsize)


def maximizeWindow():
    """ Experimental function to try to maximize a plot.

    - Tries as well as possible to maximize the figure.
    - Cf. https://stackoverflow.com/q/12439588/

    .. warning:: This function is still experimental, but "it works on my machine" so I keep it.
    """
    # plt.show(block=True)
    # plt.tight_layout()
    figManager = plt.get_current_fig_manager()
    try:
        figManager.window.showMaximized()
    except Exception:
        try:
            figManager.frame.Maximize(True)
        except Exception:
            try:
                figManager.window.state('zoomed')  # works fine on Windows!
            except Exception:
                try:
                    figManager.full_screen_toggle()
                except Exception:
                    print("  Note: Unable to maximize window...")
                    # plt.show()


#: List of formats to use for saving the figures, by default.
#: It is a smart idea to save in both a raster and vectorial formats
FORMATS = ('png', 'pdf')
# FORMATS = ('png', 'pdf', 'eps')
# FORMATS = ('png', 'pdf', 'eps', 'svg')


def show_and_save(showplot=True, savefig=None, formats=FORMATS, pickleit=False, fig=None):
    """ Maximize the window if need to show it, save it if needed, and then show it or close it.

    - Inspired by https://tomspur.blogspot.fr/2015/08/publication-ready-figures-with.html#Save-the-figure
    """
    if showplot:
        maximizeWindow()
    if savefig is not None:
        if pickleit and fig is not None:
            form = "pickle"
            path = "{}.{}".format(savefig, form)
            print("Saving raw figure with format {}, to file '{}'...".format(form, path))  # DEBUG
            with open(path, "bw") as f:
                pickle_dump(fig, f)
            print("       Saved! '{}' created of size '{}b', at '{:%c}' ...".format(path, os.path.getsize(path), datetime.fromtimestamp(os.path.getatime(path))))
        for form in formats:
            path = "{}.{}".format(savefig, form)
            print("Saving figure with format {}, to file '{}'...".format(form, path))  # DEBUG
            try:
                plt.savefig(path, bbox_inches=BBOX_INCHES)
                print("       Saved! '{}' created of size '{}b', at '{:%c}' ...".format(path, os.path.getsize(path), datetime.fromtimestamp(os.path.getatime(path))))
            except Exception as exc:
                print("Error: could not save current figure to {} because of error {}... Skipping!".format(path, exc))  # DEBUG
    try:
        plt.show(block=True) if showplot else plt.close()
    except (TypeError, AttributeError):
        print("Failed to show the figure for some unknown reason...")  # DEBUG


def add_percent_formatter(which="xaxis", amplitude=1.0, oldformatter="%.2g%%", formatter="{x:.1%}"):
    """ Small function to use a Percentage formatter for xaxis or yaxis, of a certain amplitude.

    - which can be "xaxis" or "yaxis",
    - amplitude is a float, default to 1.

    - More detail at http://stackoverflow.com/a/36320013/
    - Not that the use of matplotlib.ticker.PercentFormatter require matplotlib >= 2.0.1
    - But if not available, use matplotlib.ticker.StrMethodFormatter("{:.0%}") instead
    """
    # Which axis to use ?
    if which == "xaxis":
        ax = plt.axes().xaxis
    elif which == "yaxis":
        ax = plt.axes().yaxis
    else:
        raise ValueError("Unknown value '{}' for 'which' in function add_percent_formatter() : only xaxis,yaxis are accepted...".format(which))
    # Which formatter to use ?
    try:
        my_frmt = mtick.StrMethodFormatter(formatter)  # Use new format string
    except Exception:
        my_frmt = mtick.FormatStrFormatter(oldformatter)  # Use old format string, better looking but not correctly scaled
    if hasattr(mtick, 'PercentFormatter'):
        my_frmt = mtick.PercentFormatter(amplitude)
    # Use it!
    ax.set_major_formatter(my_frmt)


#: Default value for the ``width`` parameter for :func:`wraptext` and :func:`wraplatex`.
WIDTH = 95


def wraptext(text, width=WIDTH):
    """ Wrap the text, using ``textwrap`` module, and ``width``."""
    return "\n".join(wrap(text, width=width))


def wraplatex(text, width=WIDTH):
    """ Wrap the text, for LaTeX, using ``textwrap`` module, and ``width``."""
    return "$\n$".join(wrap(text, width=width))


def nrows_ncols(N):
    """Return (nrows, ncols) to create a subplots for N plots of the good size.

    >>> for N in range(1, 22):
    ...     nrows, ncols = nrows_ncols(N)
    ...     print("For N = {:>2}, {} rows and {} cols are enough.".format(N, nrows, ncols))
    For N =  1, 1 rows and 1 cols are enough.
    For N =  2, 2 rows and 1 cols are enough.
    For N =  3, 2 rows and 2 cols are enough.
    For N =  4, 2 rows and 2 cols are enough.
    For N =  5, 3 rows and 2 cols are enough.
    For N =  6, 3 rows and 2 cols are enough.
    For N =  7, 3 rows and 3 cols are enough.
    For N =  8, 3 rows and 3 cols are enough.
    For N =  9, 3 rows and 3 cols are enough.
    For N = 10, 4 rows and 3 cols are enough.
    For N = 11, 4 rows and 3 cols are enough.
    For N = 12, 4 rows and 3 cols are enough.
    For N = 13, 4 rows and 4 cols are enough.
    For N = 14, 4 rows and 4 cols are enough.
    For N = 15, 4 rows and 4 cols are enough.
    For N = 16, 4 rows and 4 cols are enough.
    For N = 17, 5 rows and 4 cols are enough.
    For N = 18, 5 rows and 4 cols are enough.
    For N = 19, 5 rows and 4 cols are enough.
    For N = 20, 5 rows and 4 cols are enough.
    For N = 21, 5 rows and 5 cols are enough.
    """
    nrows = int(np.ceil(np.sqrt(N)))
    ncols = N // nrows
    while N > nrows * ncols:
        ncols += 1
    nrows, ncols = max(nrows, ncols), min(nrows, ncols)
    return nrows, ncols


def addTextForWorstCases(ax, n, bins, patches, rate=0.85, normed=False, fontsize=8):
    """Add some text labels to the patches of an histogram, for the last 'rate'%.

    Use it like this, to add labels for the bins in the 65% largest values n::

        >>> n, bins, patches = plt.hist(...)
        >>> addTextForWorstCases(ax, n, bins, patches, rate=0.65)
    """
    # DONE add an automatic detection of the cases where a regret was found to not be O(log(T)) to display on the histogram the count of bad cases
    assert 0 <= rate <= 1, "Error: 'rate' = {:.3g} should be in [0, 1].".format(rate)  # DEBUG
    if not isinstance(n, list) and not isinstance(n, np.ndarray):
        n = [n]
    if hasattr(patches, 'patches'):
        # assert isinstance(patches, mpl.container.BarContainer)  # DEBUG
        patches = patches.patches
    if not isinstance(patches, list):
        patches = [patches]
    max_x = max(p.xy[0] for p in patches)
    for nx, p in zip(n, patches):
        text = "{:.3%}".format(nx) if normed else "{:.3g}".format(nx)
        x, y = p.xy[0], 1.015 * nx  # 1.5% higher than the top of the patch rectangle
        # Simple detection can be if a box is for a regret larger than some fraction of T
        if nx > 0 and x > (rate * max_x):
            # print("Writing text =", text, "at x =", x, "and y =", y)  # DEBUG
            ax.text(x, y, text, fontsize=fontsize)


def myviolinplot(*args, nonsymmetrical=False, **kwargs):
    try:
        return sns.violinplot(*args, nonsymmetrical=nonsymmetrical, cut=0, inner="stick", **kwargs)
    except (TypeError, NameError):
        return sns.violinplot(*args, cut=0, inner="stick", **kwargs)


def violin_or_box_plot(data=None, labels=None, boxplot=False, **kwargs):
    """ Automatically add labels to a box or violin plot.

    .. warning:: Requires pandas (https://pandas.pydata.org/) to add the xlabel for violin plots.
    """
    if boxplot:
        return plt.boxplot(data, labels=labels, showmeans=True, meanline=True, **kwargs)
    if labels is not None:
        try:
            import pandas as pd
            dict_of_data = {
                label: column
                for label, column in zip(labels, data)
            }
            df = pd.DataFrame(dict_of_data)
            return myviolinplot(nonsymmetrical="left", data=df, orient="v", **kwargs)
        except ImportError:
            return violin_or_box_plot(data, boxplot=boxplot, **kwargs)
    return myviolinplot(nonsymmetrical="left", data=data, orient="v", **kwargs)


MAX_NB_OF_LABELS = 50  #: If more than MAX_NB_OF_LABELS labels have to be displayed on a boxplot, don't put a legend.


def adjust_xticks_subplots(ylabel=None, labels=(), maxNbOfLabels=MAX_NB_OF_LABELS):
    """Adjust the size of the xticks, and maybe change size of ylabel.

    - See https://stackoverflow.com/a/37708190/
    """
    if len(labels) >= maxNbOfLabels:
        return
    max_length_of_labels = max([len(label) for label in labels])
    locs, xticks_labels = plt.xticks()  # XXX don't name xticks_labels, labels or it erases the argument of the function and labels are not correctly displayed.
    plt.xticks(locs, labels, rotation=80, verticalalignment="top", fontsize="xx-small")
    if max_length_of_labels >= 50:
        plt.subplots_adjust(bottom=max_length_of_labels/135.0)
        if ylabel is not None: plt.ylabel(ylabel, fontsize="x-small")
    else:
        plt.subplots_adjust(bottom=max_length_of_labels/90.0)


def table_to_latex(mean_data, std_data=None,
        labels=None, fmt_function=None, name_of_table=None,
        filename=None, erase_output=False,
        *args, **kwargs
    ):
    """ Tries to print the data from the input array or collection of array or :class:`pandas.DataFrame` to the stdout and to the file ``filename`` (if it does not exist).

    - Give ``std_data`` to print ``mean +- std`` instead of just ``mean`` from ``mean_data``,
    - Give a list to ``labels`` to use a header of the table,
    - Give a formatting function to ``fmt_function``, like :func:`IPython.core.magics.execution._format_time` to print running times, or :func:`memory_consumption.sizeof_fmt` to print memory usages, or ``lambda s: "{:.3g}".format(s)`` to print ``float`` values (default),
    - Uses :func:`tabulate.tabulate` (https://bitbucket.org/astanin/python-tabulate/) or :func:`pandas.DataFrame.to_latex` (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_latex.html#pandas.DataFrame.to_latex).

    .. warning:: FIXME this is still experimental! And useless, most of the time we simply do a copy/paste from the terminal to the LaTeX in the article...
    """
    if fmt_function is None:  fmt_function = lambda s: "{:.3g}".format(s)
    output_string = None
    input_data = mean_data
    if std_data is not None:
        format_data = np.vectorize(lambda xi, yi: r"{} \pm {}".format(fmt_function(xi), fmt_function(yi)))
        input_data = format_data(mean_data, std_data)
    else:
        format_data = np.vectorize(fmt_function)
        input_data = format_data(mean_data)
    print("Using input_data of shape = {} and size = {}\n{}".format(np.shape(input_data), np.size(input_data), input_data))  # DEBUG
    # 1. try with pandas module
    try:
        import pandas as pd
        if labels is not None:
            df = pd.DataFrame(input_data, columns=labels)
        else:
            df = pd.DataFrame(input_data)
        output_string = df.to_latex(*args, **kwargs)
    except ImportError:
        print("Error: the pandas module is not available, install it with 'pip install pandas' or 'conda install pandas'.")  # DEBUG
    # 2. if pandas failed, try with tabulate
    if output_string is None:
        try:
            import tabulate
            if labels is not None:
                output_string = tabulate.tabulate(input_data, tablefmt="latex_raw", headers=labels, *args, **kwargs)
            else:
                output_string = tabulate.tabulate(input_data, tablefmt="latex_raw", *args, **kwargs)
        except ImportError:
            print("Error: the tabulate module is not available, install it with 'pip install tabulate' or 'conda install tabulate'.")  # DEBUG
    if filename is not None and not erase_output and os.path.exists(filename):
        print("Error: the file named '{}' already exists, and option 'erase_output' is False.".format(filename))
        return -1
    if name_of_table is not None:
        output_string = r"""%% LaTeX code for a table, produced by SMPyBandits.Environment.plotsetting.table_to_latex()
\begin{table}
%s
\caption{%s}
\end{table}""" % (output_string, name_of_table)
    print("\nThe data from object (shape = {} and size = {}) can be pretty printed in a LaTeX table looking like this one:".format(np.shape(input_data), np.size(input_data)))  # DEBUG
    print(output_string)
    if filename is not None:
        print("\nThe data from object (shape = {} and size = {}) will be saved to the file {}...".format(np.shape(input_data), np.size(input_data), filename))  # DEBUG
        with open(filename, 'w') as open_file:
            print(output_string, file=open_file)
    return 0


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
