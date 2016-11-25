# -*- coding: utf-8 -*-
""" Experimental function to try to maximize a plot.

Cf. https://stackoverflow.com/q/12439588/
"""
from __future__ import print_function
import matplotlib.pyplot as plt
__author__ = "Lilian Besson"
__version__ = "0.1"


def maximizeWindow():
    """ Tries as well as possible to maximize the figure."""
    # print("Calling 'plt.tight_layout()' ...")  # DEBUG
    # plt.show()
    plt.tight_layout()
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
