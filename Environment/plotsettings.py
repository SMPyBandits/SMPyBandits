# -*- coding: utf-8 -*-
""" plotsettings: use it like this:

>>> from .plotsettings import DPI, signature, maximizeWindow, palette, makemarkers
"""
from __future__ import print_function, division

__author__ = "Lilian Besson"
__version__ = "0.6"

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Customize here if you want a signature on the titles of each plot
signature = "\n(By Lilian Besson, Fev.2017 - Code on https://Naereen.GitHub.io/AlgoBandits)"
# signature = ""  # FIXME revert to ↑ after having generating the figures for the paper

DPI = 110
FIGSIZE = (19.80, 10.80)  # in inches!
HLS = True

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


def palette(nb, hls=HLS):
    """ Use a smart palette from seaborn, for nb different plots on the same figure.

    - Ref: http://seaborn.pydata.org/generated/seaborn.hls_palette.html#seaborn.hls_palette
    """
    return sns.hls_palette(nb + 1)[:nb] if hls else sns.husl_palette(nb + 1)[:nb]


def makemarkers(nb):
    """ Give a list of cycling markers. See http://matplotlib.org/api/markers_api.html """
    allmarkers = ['o', 'v', '^', '<', '>', 'D', '*']
    longlist = allmarkers * (1 + int(nb / float(len(allmarkers))))  # Cycle the good number of time
    return longlist[:nb]  # Truncate


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


def add_percent_formatter(which="xaxis", amplitude=1.0):
    """Small function to use a Percentage formatter for xaxis or yaxis, of a certain amplitude.

    - which can be "xaxis" or "yaxis"
    - amplitude is a float, default to 1

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
    my_frmt = mtick.StrMethodFormatter("{x:.0%}")
    if hasattr(mtick, 'PercentFormatter'):
        my_frmt = mtick.PercentFormatter(amplitude)
    # Use it!
    ax.set_major_formatter(my_frmt)
