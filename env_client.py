#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client to play multi-armed bandits problem against.

Usage:
    client.py [--port=<PORT>] [--host=<HOST>] <json_configuration>
    client.py (-h|--help)
    client.py --version

Options:
    -h --help   Show this screen.
    --version   Show version.
    --port=<PORT>   Port to use for the TCP connection [default: 10000].
    --host=<HOST>   Address to use for the TCP connection [default: 0.0.0.0].
"""
from __future__ import print_function, division

__author__ = "Lilian Besson"
__version__ = "0.7"
version = "MAB environment client v{}".format(__version__)

import json
import socket
import time
from docopt import docopt

from Environment import MAB
from Arms import *


#: Example of configuration to pass from the command line.
#: ``'{"arm_type": "Bernoulli", "params": (0.1, 0.5, 0.9), "speed": 1}'``
default_configuration = {
        "arm_type": "Bernoulli",
        "params": {
            (0.1, 0.5, 0.9)
        },
        "speed": 1
    }


def read_configuration_env(a_string):
    """ Return a valid configuration dictionary to initialize a MAB environment, from the input string."""
    obj = json.loads(a_string)
    assert isinstance(obj, dict) and "arm_type" in obj and "params" in obj and "speed" in obj, "Error: invalid string to be converted to a configuration object for a MAB environment."
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
            send(sock, message)
            print("Sleeping for {} second(s)...".format(speed))
            time.sleep(speed)
    finally:
        # Clean up the socket
        print("Closing socket...")
        sock.close()


def main(arguments):
    """
    Take arguments, construct the learning policy and starts the server.
    """
    host = arguments['--host']
    port = int(arguments['--port'])
    json_configuration = arguments['<json_configuration>']
    configuration = read_configuration_env(json_configuration)
    # try to map strings in the dictionary to variables, e.g., policies
    for (key, value) in configuration.items():
        if value in globals():
            configuration[key] = globals()[value]
    # configuration['arm_type'] = globals()[configuration['arm_type']]
    env = MAB(configuration)
    print("Using the environment: ", env)  # DEBUG
    speed = float(configuration['speed'])
    print("Emitting regularly every", speed, "seconds.")  # DEBUG
    return client(env, host, port, speed)


if __name__ == '__main__':
    arguments = docopt(__doc__, version=version)
    # print("arguments =", arguments)  # DEBUG
    main(arguments)