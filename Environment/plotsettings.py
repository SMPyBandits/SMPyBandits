# -*- coding: utf-8 -*-
""" plotsettings: use it like this, in the Evaluator folder:

>>> from .plotsettings import DPI, signature, maximizeWindow, palette, makemarkers
"""
from __future__ import print_function, division

__author__ = "Lilian Besson"
__version__ = "0.6"

from textwrap import wrap
from os.path import getsize, getatime
from datetime import datetime

import matplotlib as mpl
# mpl.use('Agg')  # XXX is it a good idea? Nope, use "export MPLBACKEND='Agg'" in your bashrc ... Cf. http://stackoverflow.com/a/4935945/ and http://matplotlib.org/faq/usage_faq.html#what-is-a-backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import seaborn as sns

# Customize here if you want a signature on the titles or xlabel, of each plot
from datetime import datetime
import locale  # See this bug, http://numba.pydata.org/numba-doc/dev/user/faq.html#llvm-locale-bug
locale.setlocale(locale.LC_TIME, 'C')
monthyear = '{:%b.%Y}'.format(datetime.today()).title()
signature = "\n(By Lilian Besson, {} - Code on https://Naereen.GitHub.io/AlgoBandits)".format(monthyear)
# signature = ""  # FIXME revert to ↑ after having generating the figures for the paper

DPI = 110
FIGSIZE = (19.80, 10.80)  # in inches!

# Customize the colormap
HLS = True
VIRIDIS = False

# Bbox in inches. Only the given portion of the figure is saved. If ‘tight’, try to figure out the tight bbox of the figure.
BBOX_INCHES = 'tight'
BBOX_INCHES = None

if __name__ != '__main__':
    # use a clever color palette, eg http://seaborn.pydata.org/api.html#color-palettes
    sns.set(context="talk", style="darkgrid", palette="hls" if HLS else "husl", font="sans-serif", font_scale=1.05)

    # Use tex by default http://matplotlib.org/2.0.0/users/dflt_style_changes.html#math-text
    # mpl.rcParams['text.usetex'] = True  # XXX force use of LaTeX
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams['font.sans-serif'] = "DejaVu Sans"
    mpl.rcParams['mathtext.fontset'] = "cm"
    mpl.rcParams['mathtext.rm'] = "serif"

    # Configure the DPI of all images, once and for all!
    mpl.rcParams['figure.dpi'] = DPI
    print(" - Setting dpi of all figures to", DPI, "...")

    # Configure figure size, even of if saved directly and not displayed, use HD screen
    # cf. https://en.wikipedia.org/wiki/Computer_display_standard
    mpl.rcParams['figure.figsize'] = FIGSIZE
    print(" - Setting 'figsize' of all figures to", FIGSIZE, "...")

    # Set up a discrete version of the Viridis map for axes.prop_cycle

    # # Check that a XServer is available
    # fig = plt.figure()
    # fig.close()


def palette(nb, hls=HLS, viridis=VIRIDIS):
    """ Use a smart palette from seaborn, for nb different plots on the same figure.

    - Ref: http://seaborn.pydata.org/generated/seaborn.hls_palette.html#seaborn.hls_palette

    >>> sns.palplot(palette(10, hls=True))
    >>> sns.palplot(palette(10, hls=False))  # use HUSL by default
    >>> sns.palplot(palette(10, viridis=True))
    """
    if viridis:
        return sns.color_palette('viridis', nb)
    else:
        return sns.hls_palette(nb + 1)[:nb] if hls else sns.husl_palette(nb + 1)[:nb]


def makemarkers(nb):
    """ Give a list of cycling markers. See http://matplotlib.org/api/markers_api.html """
    allmarkers = ['o', 'v', '^', '<', '>', 'D', '*']
    longlist = allmarkers * (1 + int(nb / float(len(allmarkers))))  # Cycle the good number of time
    return longlist[:nb]  # Truncate


def legend():
    """plt.legend() with good options, cf. http://matplotlib.org/users/recipes.html#transparent-fancy-legends."""
    plt.legend(loc='best', numpoints=1, fancybox=True, framealpha=0.8)


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


def show_and_save(showplot=True, savefig=None, formats=('png', 'pdf')):
    """Maximize the window, save it if needed, and then show it or close it.

    - Inspired by https://tomspur.blogspot.fr/2015/08/publication-ready-figures-with.html#Save-the-figure
    """
    maximizeWindow()
    if savefig is not None:
        for form in formats:
            path = "{}.{}".format(savefig, form)
            print("Saving figure with format {}, to file '{}'...".format(form, path))  # DEBUG
            plt.savefig(path, bbox_inches=BBOX_INCHES)
            print("       Saved! '{}' created of size '{}b', at '{:%c}' ...".format(path, getsize(path), datetime.fromtimestamp(getatime(path))))
    plt.show() if showplot else plt.close()


def add_percent_formatter(which="xaxis", amplitude=1.0, oldformatter='%.2g%%', formatter='{x:.1%}'):
    """Small function to use a Percentage formatter for xaxis or yaxis, of a certain amplitude.

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
    except:
        my_frmt = mtick.FormatStrFormatter(oldformatter)  # Use old format string, better looking but not correctly scaled
    if hasattr(mtick, 'PercentFormatter'):
        my_frmt = mtick.PercentFormatter(amplitude)
    # Use it!
    ax.set_major_formatter(my_frmt)


def wraptext(text, width=110):
    """Wrap the text, using textwrap module, and width."""
    return '\n'.join(wrap(text, width=width))


def wraplatex(text, width=110):
    """Wrap the text, for LaTeX, using textwrap module, and width."""
    return '$\n$'.join(wrap(text, width=width))
