#! /usr/bin/env python2
# -*- coding: utf-8; mode: python -*-
""" Small experimental bot for our Slack team at SCEE (https://sceeteam.slack.com/), CentraleSupélec campus de Rennes.

It reads a file full of quotes (from TV shows), and post one randomly at random times on the channel #random.

Requirements:
- slackclient is required
- If progressbar (https://pypi.python.org/pypi/progressbar) is installed, use it.

About:
- *Date:* 13/02/2017.
- *Author:* Lilian Besson, (C) 2017
- *Licence:* MIT Licence (http://lbesson.mit-license.org).
"""

from __future__ import division, print_function  # Python 2 compatibility

import sys
import os
import random
from os.path import join, expanduser
import time

import logging
logging.basicConfig(
    format="%(asctime)s  %(levelname)s: %(message)s",
    datefmt='%m-%d-%Y %I:%M:%S %p',
    level=logging.INFO
)

from numpy.random import poisson
from slackclient import SlackClient

# Import algorithms
from Policies import Thompson

# --- Parameters of the bot

MINUTES = 60
HOURS = 60 * MINUTES

QUOTE_FILE = os.getenv("quotes", expanduser(join("~", ".quotes.txt")))

SLACK_TOKEN = open(expanduser(join("~", ".slack_api_key")), 'r').readline().strip()

USE_CHANNEL = False  # DEBUG
USE_CHANNEL = True

DONTSEND = True  # DEBUG
DONTSEND = False

SLACK_USER = "@lilian"
SLACK_CHANNEL = "#random"
SLACK_CHANNEL = "#test"

DEFAULT_CHANNEL = SLACK_CHANNEL if USE_CHANNEL else SLACK_USER

MEAN_TIME = (1 * HOURS) if USE_CHANNEL else 60

URL = "https://bitbucket.org/lbesson/bin/src/master/my-small-slack-bot.py"

POSITIVE_REACTIONS = ['up', '+1', 'thumbsup']
NEGATIVES_REACTIONS = ['down', '-1', 'thumbsdown']


# --- Functions

def sleeptime(lmbda=MEAN_TIME, use_poisson=True):
    """Random time until next message."""
    if use_poisson:
        return poisson(lmbda)
    else:
        return lmbda


def sleep_bar(secs):
    """Sleep with a bar, or not"""
    try:
        # From progressbar example #3, https://github.com/niltonvolpato/python-progressbar/blob/master/examples.py#L67
        from progressbar import Bar, ETA, ProgressBar, ReverseBar
        widgets = [Bar('>'), ' ', ETA(), ' ', ReverseBar('<')]
        pbar = ProgressBar(widgets=widgets, maxval=100).start()
        for i in range(100):
            # do something
            time.sleep(secs / 110.)
            pbar.update(i)
        pbar.finish()
    except ImportError:
        time.sleep(secs)


def random_line(lines):
    """Read the file and select one line."""
    try:
        return random.choice(lines).replace('`', '').replace('_', '')
    except Exception as e:  # Default quote
        logging.info("Failed to read a random line from this list with {} lines...".format(len(lines)))  # DEBUG
        return "I love you !"


def print_TS(TS, name_channels):
    print("For this Thompson sampling algorithm ({}), the posteriors on channels are currently:".format(TS))
    for arm in range(TS.nbArms):
        print(" - For arm #{:^2} named {:^15} : posterior = {} ...".format(arm, name_channels[arm], TS.posterior[arm]))


# --- API calls

def get_list_channels(sc):
    """Get list of channels."""
    # https://api.slack.com/methods/channels.list
    response = sc.api_call(
        "channels.list",
    )
    return response['channels']


def get_full_list_channels(sc, list_channels=None):
    """Get list of channels."""
    if list_channels is None:
        list_channels = get_list_channels(sc)
    full_list_channels = []
    for c in list_channels:
        # https://api.slack.com/methods/channels.info
        response = sc.api_call(
            "channels.info", channel=c['id']
        )
        if response['ok']:
            full_list_channels.append(response['channel'])
    return full_list_channels


def choose_channel(list_channels):
    """Any method to choose the channels. Pure random currently."""
    return random.choice(list_channels)['name']


def most_popular_channel(list_channels):
    """Any method to choose the channels. Choose the more crowded."""
    nums_names = [(c['num_members'], c['name']) for c in list_channels]
    maxnum = sorted(nums_names, reverse=True)[0][0]
    return random.choice([c[1] for c in nums_names if c[0] == maxnum])


def last_read_channel(sc, full_list_channels):
    """Any method to choose the channels. Choose the last read one."""
    nums_names = [(c['last_read'], c['name']) for c in full_list_channels if 'last_read' in c]
    maxnum = sorted(nums_names, reverse=True)[0][0]
    return random.choice([c[1] for c in nums_names if c[0] == maxnum])


def get_reactions(list_of_ts_channel, sc):
    """Get the reaction of users on all the messages sent by the bot, to increase or decrease the frequency of messages."""
    scale_factor = 1.
    try:
        for (ts, c) in list_of_ts_channel:
            # https://api.slack.com/methods/reactions.get
            reaction = sc.api_call(
                "reactions.get", channel=c, timestamp=ts
            )
            logging.debug("reaction =", reaction)
            if 'message' not in reaction:
                continue
            text = {t['name']: t['count'] for t in reaction['message']['reactions']}
            logging.info("text =", text)
            if any(s in text.keys() for s in POSITIVE_REACTIONS):
                nb = max([0.5] + [text[s] for s in POSITIVE_REACTIONS if s in text.keys()])
                logging.info("I read {} positive reactions ...".format(int(nb)))
                scale_factor /= 2 * nb
            elif any(s in text for s in NEGATIVES_REACTIONS):
                nb = max([0.5] + [text[s] for s in NEGATIVES_REACTIONS if s in text.keys()])
                logging.info("I read {} negative reactions ...".format(int(nb)))
                scale_factor *= 2 * nb
            elif "rage" in text:
                raise ValueError("One user reacted with :rage:, the bot will quit...")
        return scale_factor
    except KeyError:
        return scale_factor


def send(text, sc, channel=DEFAULT_CHANNEL, dontsend=False):
    """Send text to channel SLACK_CHANNEL with client sc.

    - https://github.com/slackapi/python-slackclient#sending-a-message
    """
    text = "{}\n> (Sent by an _open-source_ Python script :snake:, {}, written by Lilian Besson)".format(text, URL)
    logging.info("{}ending the message '{}' to channel/user '{}' ...".format("NOT s" if dontsend else "S", text, channel))
    if dontsend:
        return {}  # Empty response
    else:
        # https://api.slack.com/methods/chat.postMessage
        response1 = sc.api_call(
            "chat.postMessage", channel=channel, text=text,
            username="Citations aléatoires", icon_emoji=":robot_face:"
        )
        ts = response1['ts']
        response2 = sc.api_call(
            "pins.add", channel=channel, ts=ts
        )
        return response1


def has_been_sent(sc, t0, t1, ts, c):
    # https://api.slack.com/methods/chat.postMessage
    response = sc.api_call(
        "channels.history", channel=c, oldest=t0, latest=t1,
    )
    # print(response)  # DEBUG
    return response['ok'] and 0 < len([True for m in response['messages'] if m['ts'] == ts])


def has_been_seen(sc, ts, c):
    # https://api.slack.com/methods/chat.postMessage
    response = sc.api_call(
        "channels.info", channel=c,
    )
    # print(response['channel']['last_read'])  # DEBUG
    # print("ts =", ts)  # DEBUG
    return response['ok'] and ts <= response['channel']['last_read']


def loop(quote_file=QUOTE_FILE, random_channel=False, dontsend=False):
    """Main loop."""
    logging.info("Starting my Slack bot, reading random quotes from the file {}...".format(quote_file))
    # Get list of quotes and parameters
    the_quote_file = open(quote_file, 'r')
    lines = the_quote_file.readlines()
    lmbda = MEAN_TIME
    # API
    sc = SlackClient(SLACK_TOKEN)
    list_channels = get_list_channels(sc)
    name_channels = [c['name'] for c in list_channels]
    full_list_channels = get_full_list_channels(sc, list_channels)
    nb_channels = len(list_channels)
    # Thompson sampling algorithms
    TS = Thompson(nb_channels)
    TS.startGame()
    # Start loop
    list_of_ts_channel = []
    while True:
        # 1. get random quote
        text = random_line(lines)
        # logging.info("New message:\n{}".format(text))
        if random_channel:
            # channel1 = choose_channel(list_channels)
            # print("- method choose_channel gave channel1 =", channel1)  # DEBUG
            # channel2 = most_popular_channel(list_channels)
            # print("- method most_popular_channel gave channel2 =", channel2)  # DEBUG
            # channel3 = last_read_channel(sc, full_list_channels)
            # print("- method last_read_channel gave channel3 =", channel3)  # DEBUG
            channel4 = list_channels[TS.choice()]['name']
            # print("- method TS.choice gave channel3 =", channel4)  # DEBUG
            channel = random.choice([SLACK_CHANNEL, SLACK_USER, channel4])
            # print("- method random.choice gave channel =", channel)  # DEBUG
            # channel = SLACK_CHANNEL
            channel = channel4
        else:
            channel = DEFAULT_CHANNEL
        response = send(text, sc, channel=channel, dontsend=dontsend)
        # 2. sleep until next quote
        secs = sleeptime(lmbda)
        str_secs = time.asctime(time.localtime(time.time() + secs))
        logging.info("  ... Next message in {} seconds, at {} ...".format(secs, str_secs))
        sleep_bar(secs)
        # 3. get response
        try:
            ts, c = response['ts'], response['channel']
            list_of_ts_channel.append((ts, c))
            # Get reaction from users on the messages already posted
            scale_factor = get_reactions(list_of_ts_channel, sc)
            lmbda = scale_factor * MEAN_TIME  # Don't accumulate this!
        except KeyError:
            pass
        logging.info("  Currently, the mean time between messages is {} ...".format(lmbda))
        # 4. try to know if the last message was seen, to give a reward to the TS algorithm
        try:
            ts, c = response['ts'], response['channel']
            # response = int(has_been_sent(sc, ts-60*60, ts+10+secs, ts, c))
            reward = int(has_been_seen(sc, ts, c))
        except KeyError:
            reward = 0  # not seen, by default
        cnl = channel.replace('#', '')
        ch = name_channels.index(cnl) if cnl in name_channels else -1
        if ch >= 0:
            print("Giving reward", reward, "for channel", ch, "to Thompson algorithm ...")
            # 5. Print current posteriors
            TS.getReward(ch, reward)
            print_TS(TS, name_channels)
        else:
            print("reward =", reward)  # DEBUG
            print("cnl =", cnl)  # DEBUG
    return 0


# --- Main script

if __name__ == '__main__':
    quote_file = sys.argv[1] if len(sys.argv) > 1 else QUOTE_FILE
    sys.exit(loop(quote_file, random_channel=True, dontsend=DONTSEND))

# End of my-small-slack-bot.py
