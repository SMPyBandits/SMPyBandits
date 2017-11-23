#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client to play multi-armed bandits problem against.
Many distribution of arms are supported, default to Bernoulli.

Usage:
    env_client.py [markovian | dynamic] [--port=<PORT>] [--host=<HOST>] [--speed=<SPEED>] <json_configuration>
    env_client.py (-h|--help)
    env_client.py --version

Options:
    -h --help   Show this screen.
    --version   Show version.
    markovian   Whether to use a Markovian MAB problem (default is simple MAB problems).
    dynamic     Whether to use a Dynamic MAB problem (default is simple MAB problems).
    --port=<PORT>   Port to use for the TCP connection [default: 10000].
    --host=<HOST>   Address to use for the TCP connection [default: 0.0.0.0].
    --speed=<SPEED>   Speed of emission in milliseconds [default: 1000].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.7"
version = "MAB environment client v{}".format(__version__)

import json
import socket
import time
from docopt import docopt

from Environment import MAB, MarkovianMAB, DynamicMAB
from Arms import *


#: Example of configuration to pass from the command line.
#: ``'{"arm_type": "Bernoulli", "params": (0.1, 0.5, 0.9)}'``
default_configuration = {
        "arm_type": "Bernoulli",
        "params": {
            (0.1, 0.5, 0.9)
        }
    }


def read_configuration_env(a_string):
    """ Return a valid configuration dictionary to initialize a MAB environment, from the input string."""
    obj = json.loads(a_string)
    assert isinstance(obj, dict) and "arm_type" in obj and "params" in obj, "Error: invalid string to be converted to a configuration object for a MAB environment."
    return obj


def send_message(sock, message):
    # Send data
    print("sending {!r}".format(message))
    sock.sendall(message)


def client(env, host, port, speed):
    """
    Launch an client that:

    - uses sockets to listen to input and reply
    - create a MAB environment from a JSON configuration (exactly like ``main.py`` when it reads ``configuration.py``)
    - then receives choice ``arm`` from the network, pass it to the MAB environment, listens to his ``reward = draw(arm)`` feedback, and sends this back to the network.
    """
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = (host, port)
    print("starting up on {} port {}".format(*server_address))
    sock.connect(server_address)
    t = -1

    try:
        message = "Hi!".encode()
        print("\nSending first message = {!r}".format(message))
        send_message(sock, message)
        print("Sleeping for {} second(s)...".format(speed))
        time.sleep(speed)
        while True:
            t += 1
            arm = t % env.nbArms

            data = sock.recv(16)
            message = data.decode()
            print("\nData received: {!r}".format(message))
            arm = int(message)

            reward = env.draw(arm)
            message = str(reward)[:7].encode()
            print("Environment = {}, at time t = {}:".format(env, t))
            print("Sending random reward = {!r} from arm {} ...".format(message, arm))
            send_message(sock, message)
            print("Sleeping for {} second(s)...".format(speed))
            time.sleep(speed)
    finally:
        # Clean up the socket
        print("Closing socket...")
        sock.close()


def transform_str(params):
    """Like a safe exec() on a dictionary that can contain special values:

    - strings are interpreted as variables names (e.g., policy names) from the current ``globals()`` scope,
    - list are transformed to tuples to be constant and hashable,
    - dictionary are recursively transformed.
    """
    for (key, value) in params.items():
        if isinstance(value, dict):
            transform_str(value)
        if isinstance(value, list):  # unhashable
            value = tuple(value)
        try:
            if value in globals():
                params[key] = globals()[value]
        except TypeError:
            pass


def main(arguments):
    """
    Take arguments, construct the learning policy and starts the server.
    """
    host = arguments['--host']
    port = int(arguments['--port'])
    speed = float(arguments['--speed']) / 1000.0
    is_markovian = arguments['markovian']
    is_dynamic = arguments['dynamic']
    json_configuration = arguments['<json_configuration>']
    configuration = read_configuration_env(json_configuration)
    transform_str(configuration)
    if is_dynamic:
        env = DynamicMAB(configuration)
    elif is_markovian:
        env = MarkovianMAB(configuration)
    else:
        env = MAB(configuration)
    print("Using the environment: ", env)  # DEBUG
    print("Emitting regularly every", speed, "seconds.")  # DEBUG
    return client(env, host, port, speed)


if __name__ == '__main__':
    arguments = docopt(__doc__, version=version)
    # print("arguments =", arguments)  # DEBUG
    main(arguments)