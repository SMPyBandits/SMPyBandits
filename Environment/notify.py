#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines one useful function :func:`notify()` to (try to) send a desktop notification.

- Only tested on Ubuntu and Debian desktops.
- Should work on any FreeDesktop compatible desktop, see https://wiki.ubuntu.com/NotifyOSD.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from os import getcwd
from os.path import exists, join
from subprocess import Popen


VERBOUS = False

# Constants for the program
PROGRAM_NAME = "SMPyBandits"  #: Program name

# ICON_PATH = join("..", "logo.png")  #: Icon to use
ICON_PATH = "logo.png"  #: Icon to use


# Define the icon loaded function
try:
    from gi import require_version
    require_version('GdkPixbuf', '2.0')
    from gi.repository import GdkPixbuf

    def load_icon():
        """ Load and open the icon. """
        # Loading the icon...
        if exists(ICON_PATH):
            # Use GdkPixbuf to create the proper image type
            iconpng = GdkPixbuf.Pixbuf.new_from_file(ICON_PATH)
        else:
            iconpng = None
        # print("iconpng =", iconpng)  # DEBUG
        return iconpng

except ImportError:
    if VERBOUS:
        print("\nError, gi.repository.GdkPixbuf seems to not be available, so notification icons will not be available ...")
        print("On Ubuntu, if you want notification icons to work, install the 'python-gobject' and 'libnotify-bin' packages.")
        print("(For more details, cf. 'http://www.devdungeon.com/content/desktop-notifications-python-libnotify')")

    def load_icon():
        """ Load and open the icon. """
        return None


#: Trying to import gi.repository.Notify
has_Notify = False
try:
    from gi import require_version
    require_version('Notify', '0.7')
    from gi.repository import Notify
    # One time initialization of libnotify
    Notify.init(PROGRAM_NAME)
    has_Notify = True
except ImportError:
    if VERBOUS:
        print("\nError, gi.repository.Notify seems to not be available, so notification will not be available ...")
        print("On Ubuntu, if you want notifications to work, install the 'python-gobject' and 'libnotify-bin' packages.")
        print("(For more details, cf. 'http://www.devdungeon.com/content/desktop-notifications-python-libnotify')")


# Define the first notify function, with gi.repository.Notify
def notify_gi(body, summary=PROGRAM_NAME, icon="terminal",
              timeout=5  # In seconds
              ):
    """ Send a notification, with gi.repository.Notify.

    - icon can be "dialog-information", "dialog-warn", "dialog-error".
    """
    try:
        # Trying to fix a bug:
        # g-dbus-error-quark: GDBus.Error:org.freedesktop.DBus.Error.ServiceUnknown: The name :1.5124 was not provided by any .service files (2)
        Notify.init(PROGRAM_NAME)
        # XXX maybe the PROGRAM_NAME should be random?

        # Cf. http://www.devdungeon.com/content/desktop-notifications-python-libnotify
        # Create the notification object
        notification = Notify.Notification.new(
            summary,  # Title of the notification
            body,     # Optional content of the notification
            icon      # XXX Should not indicate it here
        )

        # Lowest urgency (LOW, NORMAL or CRITICAL)
        notification.set_urgency(Notify.Urgency.LOW)

        # add duration, lower than 10 seconds (5 second is enough).
        notification.set_timeout(timeout * 1000)

        # Actually show the notification on screen
        notification.show()
        return 0

    # Ugly! XXX Catches too general exception
    except Exception as e:
        print("\nnotify.notify(): Error, notify.notify() failed, with this exception")
        print(e)
        return -1


# Define the second notify function, with a subprocess call to 'notify-send'
def notify_cli(body, summary=PROGRAM_NAME, icon="terminal",
               timeout=5  # In seconds
               ):
    """ Send a notification, with a subprocess call to 'notify-send'."""
    try:
        print("notify.notify(): Trying to use the command line program 'notify-send' ...")
        icon = join(getcwd(), icon)
        Popen(["notify-send", "--expire-time=%s" % (timeout * 1000), "--icon=%s" % icon, summary, body])
        print("notify.notify(): A notification have been sent, with summary = '%s', body = '%s', expire-time='%s' and icon='%s'." % (summary, body, timeout * 1000, icon))
        return 0
    # Ugly! XXX Catches too general exception
    except Exception as e:
        print("\nnotify.notify(): notify-send : not-found ! Returned exception is %s." % e)
        return -1


# Define the unified notify.notify() function
def notify(body, summary=PROGRAM_NAME, icon="terminal",
           timeout=5  # In seconds
           ):
    """ Send a notification, using one of the previously defined method, until it works. Usually it works."""
    # print("Notification: '{}', from '{}' with icon '{}'.".format(body, summary, icon))  # DEBUG
    if not has_Notify:
        print("notify.notify(): Warning, desktop notification from Python seems to not be available ...")
        return notify_cli(body, summary=summary, icon=icon, timeout=timeout)
    else:
        try:
            return_code = notify_gi(body, summary=summary, icon=icon, timeout=timeout)
            if return_code < 0:
                return_code = notify_cli(body, summary=summary, icon=icon, timeout=timeout)
        except Exception:
            return_code = notify_cli(body, summary=summary, icon=icon, timeout=timeout)
        return return_code


if __name__ == "__main__":
    notify("Test body Test body Test body ! From 'notify(...)' with icon=terminal ...", icon="terminal")
