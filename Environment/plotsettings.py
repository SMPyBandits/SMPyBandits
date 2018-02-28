# -*- coding: utf-8 -*-
""" plotsettings: use it like this, in the Environment folder:

>>> from .plotsettings import BBOX_INCHES, signature, maximizeWindow, palette, makemarkers, add_percent_formatter, wraptext, wraplatex, legend, show_and_save, nrows_ncols
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from textwrap import wrap
from os.path import getsize, getatime

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
monthyear = '{:%b.%Y}'.format(datetime.today()).title()  #: Month.Year date

from os import getenv

# Backup figure objects
import pickle

if getenv('DEBUG', 'False') == 'True':
    signature = "\n(By Lilian Besson, {} - Code on https://github.com/SMPyBandits/SMPyBandits - MIT Licensed)".format(monthyear)  #: A small string to use as a signature
else:
    signature = ""

DPI = 120  #: DPI to use for the figures
FIGSIZE = (19.80, 10.80)  #: Figure size, in inches!

# Customize the colormap
HLS = True  #: Use the HLS mapping, or HUSL mapping
VIRIDIS = False  #: Use the Viridis colormap

# Bbox in inches. Only the given portion of the figure is saved. If 'tight', try to figure out the tight bbox of the figure.
BBOX_INCHES = 'tight'  #: Use this parameter for bbox
BBOX_INCHES = None

if __name__ != '__main__':
    # use a clever color palette, eg http://seaborn.pydata.org/api.html#color-palettes
    sns.set(context="talk", style="whitegrid", palette="hls" if HLS else "husl", font="sans-serif", font_scale=1.1)

    # Use tex by default http://matplotlib.org/2.0.0/users/dflt_style_changes.html#math-text
    # mpl.rcParams['text.usetex'] = True  # XXX force use of LaTeX
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams['font.sans-serif'] = "DejaVu Sans"
    mpl.rcParams['mathtext.fontset'] = "cm"
    mpl.rcParams['mathtext.rm'] = "serif"

    # Configure the DPI of all images, once and for all!
    mpl.rcParams['figure.dpi'] = DPI
    # print(" - Setting dpi of all figures to", DPI, "...")  # DEBUG

    # Configure figure size, even of if saved directly and not displayed, use HD screen
    # cf. https://en.wikipedia.org/wiki/Computer_display_standard
    mpl.rcParams['figure.figsize'] = FIGSIZE
    # print(" - Setting 'figsize' of all figures to", FIGSIZE, "...")  # DEBUG

    # Set up a discrete version of the Viridis map for axes.prop_cycle

    # # Check that a XServer is available
    # fig = plt.figure()
    # fig.close()


def palette(nb, hls=HLS, viridis=VIRIDIS):
    """ Use a smart palette from seaborn, for nb different plots on the same figure.

    - Ref: http://seaborn.pydata.org/generated/seaborn.hls_palette.html#seaborn.hls_palette

    >>> palette(10, hls=True)  # doctest: +ELLIPSIS
    [(0.86..., 0.37..., 0.33...),
     (0.86..., 0.65..., 0.33...),
     (0.78..., 0.86..., 0.33...),
     (0.49..., 0.86..., 0.33...),
     (0.33..., 0.86..., 0.46...),
     (0.33..., 0.86..., 0.74...),
     (0.33..., 0.68..., 0.86...),
     (0.33..., 0.40..., 0.86...),
     (0.56..., 0.33..., 0.86...),
     (0.84..., 0.33..., 0.86...)]
    >>> palette(10, hls=False)  # doctest: +ELLIPSIS
    [[0.967..., 0.441..., 0.535...],
     [0.883..., 0.524..., 0.195...],
     [0.710..., 0.604..., 0.194...],
     [0.543..., 0.654..., 0.193...],
     [0.195..., 0.698..., 0.345...],
     [0.206..., 0.682..., 0.582...],
     [0.214..., 0.671..., 0.698...],
     [0.225..., 0.653..., 0.841...],
     [0.559..., 0.576..., 0.958...],
     [0.857..., 0.440..., 0.957...]]
    >>> palette(10, viridis=True)  # doctest: +ELLIPSIS
    [(0.283..., 0.130..., 0.449...),
     (0.262..., 0.242..., 0.520...),
     (0.220..., 0.343..., 0.549...),
     (0.177..., 0.437..., 0.557...),
     (0.143..., 0.522..., 0.556...),
     (0.119..., 0.607..., 0.540...),
     (0.166..., 0.690..., 0.496...),
     (0.319..., 0.770..., 0.411...),
     (0.525..., 0.833..., 0.288...),
     (0.762..., 0.876..., 0.137...)]

    - To visualize:

    >>> sns.palplot(palette(10, hls=True))
    >>> sns.palplot(palette(10, hls=False))  # use HUSL by default
    >>> sns.palplot(palette(10, viridis=True))
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
# SHRINKFACTOR = 0.75
SHRINKFACTOR = 0.65
SHRINKFACTOR = 0.60


def legend(putatright=PUTATRIGHT, shrinkfactor=SHRINKFACTOR, fig=None):
    """plt.legend() with good options, cf. http://matplotlib.org/users/recipes.html#transparent-fancy-legends.


    - It can place the legend to the right also, see https://stackoverflow.com/a/4701285/.
    """
    if fig is None:
        hack_fig = plt.gcf()
        fig = plt  # HACK XXX WARNING
    if putatright:
        try:
            # Shrink current axis by 20% on xaxis and 10% on yaxis
            delta_rect = (1. - shrinkfactor)/5.
            # rect = [left, bottom, right, top] in normalized (0, 1) figure coordinates.
            fig.tight_layout(rect=[delta_rect/1.25, delta_rect, shrinkfactor, 1 - delta_rect])
            # Put a legend to the right of the current axis
            fig.legend(loc='center left', numpoints=1, fancybox=True, framealpha=0.8, bbox_to_anchor=(1, 0.5))
        except:
            fig.legend(loc='best', numpoints=1, fancybox=True, framealpha=0.8)
    else:
        fig.legend(loc='best', numpoints=1, fancybox=True, framealpha=0.8)


def maximizeWindow():
    """ Experimental function to try to maximize a plot.

    - Tries as well as possible to maximize the figure.
    - Cf. https://stackoverflow.com/q/12439588/
    """
    # print("Calling 'plt.tight_layout()' ...")  # DEBUG
    # plt.show()
    # plt.tight_layout()
    # print("Calling 'figManager = plt.get_current_fig_manager()' ...")  # DEBUG
    figManager = plt.get_current_fig_manager()
    try:
        # print("Calling 'figManager.window.showMaximized()' ...")  # DEBUG
        figManager.window.showMaximized()
    except Exception:
        try:
            # print("Calling 'figManager.frame.Maximize(True)' ...")  # DEBUG
            figManager.frame.Maximize(True)
        except Exception:
            try:
                # print("Calling 'figManager.window.state('zoomed')' ...")  # DEBUG
                figManager.window.state('zoomed')  # works fine on Windows!
            except Exception:
                try:
                    # print("Calling 'figManager.full_screen_toggle()' ...")  # DEBUG
                    figManager.full_screen_toggle()
                except Exception:
                    print("  Note: Unable to maximize window...")
                    # plt.show()


#: List of formats to use for saving the figures, by default.
#: It is a smart idea to save in both a raster and vectorial formats
FORMATS = ('png', 'pdf', 'eps')
# FORMATS = ('png', 'pdf', 'eps', 'svg')


def show_and_save(showplot=True, savefig=None, formats=FORMATS, pickleit=False, fig=None):
    """ Maximize the window, save it if needed, and then show it or close it.

    - Inspired by https://tomspur.blogspot.fr/2015/08/publication-ready-figures-with.html#Save-the-figure
    """
    maximizeWindow()
    if savefig is not None:
        if pickleit and fig is not None:
            form = "pickle"
            path = "{}.{}".format(savefig, form)
            print("Saving raw figure with format {}, to file '{}'...".format(form, path))  # DEBUG
            with open(path, "bw") as f:
                pickle.dump(fig, f)
            print("       Saved! '{}' created of size '{}b', at '{:%c}' ...".format(path, getsize(path), datetime.fromtimestamp(getatime(path))))
        for form in formats:
            path = "{}.{}".format(savefig, form)
            print("Saving figure with format {}, to file '{}'...".format(form, path))  # DEBUG
            plt.savefig(path, bbox_inches=BBOX_INCHES)
            print("       Saved! '{}' created of size '{}b', at '{:%c}' ...".format(path, getsize(path), datetime.fromtimestamp(getatime(path))))
    try:
        plt.show() if showplot else plt.close()
    except (TypeError, AttributeError):
        print("Failed to show the figure for some unknown reason...")  # DEBUG


def add_percent_formatter(which="xaxis", amplitude=1.0, oldformatter='%.2g%%', formatter='{x:.1%}'):
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
WIDTH = 150


def wraptext(text, width=WIDTH):
    """ Wrap the text, using ``textwrap`` module, and ``width``."""
    return '\n'.join(wrap(text, width=width))


def wraplatex(text, width=WIDTH):
    """ Wrap the text, for LaTeX, using ``textwrap`` module, and ``width``."""
    return '$\n$'.join(wrap(text, width=width))


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
    max_x = max(p.xy[0] for p in patches)
    for nx, b, p in zip(n, bins[1:], patches):
        text = "{:.3%}".format(nx) if normed else "{:.3g}".format(nx)
        x, y = p.xy[0], 1.015 * nx  # 1.5% higher than the top of the patch rectangle
        # Simple detection can be if a box is for a regret larger than some fraction of T
        if nx > 0 and x > (rate * max_x):
            # print("Writing text =", text, "at x =", x, "and y =", y)  # DEBUG
            ax.text(x, y, text, fontsize=fontsize)


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
