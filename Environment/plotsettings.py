# -*- coding: utf-8 -*-
""" plotsettings: use it like this:

>>> from .plotsettings import DPI, signature, maximizeWindow, palette, makemarkers
"""
from __future__ import print_function

__author__ = "Lilian Besson"
__version__ = "0.2"

import matplotlib.pyplot as plt
import seaborn as sns

# Customize here if you want a signature on the titles of each plot
signature = "\n(By Lilian Besson, Jan.2017 - Code on https://Naereen.GitHub.io/AlgoBandits)"
# signature = ""  # FIXME revert to ↑ after having generating the figures for the paper

DPI = 140
HLS = True

if __name__ != '__main__':
    plt.xkcd()  # XXX turn on XKCD-like style ?! cf. http://matplotlib.org/xkcd/ for more details
    # FIXED use a clever color palette, eg http://seaborn.pydata.org/api.html#color-palettes
    sns.set(context="talk",
            style="darkgrid",
            palette="hls" if HLS else "husl",
            font="sans-serif",
            font_scale=1.0
            )


def palette(nb, hls=HLS):
    """ Use a smart palette from seaborn, for nb different things to plot.

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
    try:
        # print("Calling 'figManager = plt.get_current_fig_manager()' ...")  # DEBUG
        figManager = plt.get_current_fig_manager()
        # print("Calling 'figManager.window.showMaximized()' ...")  # DEBUG
        figManager.window.showMaximized()
    except:
        try:
            # print("Calling 'figManager.frame.Maximize(True)' ...")  # DEBUG
            figManager.frame.Maximize(True)
        except:
            try:
                # print("Calling 'figManager.window.state('zoomed')' ...")  # DEBUG
                figManager.window.state('zoomed')  # works fine on Windows!
            except:
                try:
                    # print("Calling 'figManager.full_screen_toggle()' ...")  # DEBUG
                    figManager.full_screen_toggle()
                except:
                    print("Unable to maximize window...")
    # plt.show()
