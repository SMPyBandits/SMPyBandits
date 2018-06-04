#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Server to play multi-armed bandits problem against.

Usage:
    policy_server.py [--port=<PORT>] [--host=<HOST>] [--means=<MEANS>] <json_configuration>
    policy_server.py (-h|--help)
    policy_server.py --version

Options:
    -h --help       Show this screen.
    --version       Show version.
    --port=<PORT>   Port to use for the TCP connection [default: 10000].
    --host=<HOST>   Address to use for the TCP connection [default: 0.0.0.0].
    --means=<MEANS> Means of arms used by the environment, to print regret [default: None].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"
version = "SMPyBandits MAB policy server v{}".format(__version__)

import json
import socket
import numpy as np
try:
    from docopt import docopt
except ImportError:
    print("ERROR: the 'docopt' module is needed for this script 'policy_server.py'.\nPlease install it with 'sudo pip install docopt' (or pip3), and try again!\nIf the issue persists, please fill a ticket here: https://github.com/SMPyBandits/SMPyBandits/issues/new")  # DEBUG

from Policies import *


#: Example of configuration to pass from the command line.
#: ``'{"nbArms": 3, "archtype": "UCBalpha", "params": { "alpha": 0.5 }}'``
default_configuration = {
        "nbArms": 10,
        "archtype": "UCBalpha",   # This basic UCB is very worse than the other
        "params": {
            "alpha": 1,
        }
    }

def read_configuration_policy(a_string):
    """ Return a valid configuration dictionary to initialize a policy, from the input string."""
    obj = json.loads(a_string)
    assert isinstance(obj, dict) and "nbArms" in obj and "archtype" in obj and "params" in obj, "Error: invalid string to be converted to a configuration object for a policy."
    return obj


def server(policy, host, port, means=None):
    """
    Launch a server that:

    - uses sockets to listen to input and reply
    - create a learning algorithm from a JSON configuration (exactly like ``main.py`` when it reads ``configuration.py``)
    - then receives feedback ``(arm, reward)`` from the network, pass it to the algorithm, listens to his ``arm = choice()`` suggestion, and sends this back to the network.
    """
    has_index = hasattr(policy, "index")

    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = (host, port)
    print("starting up on {} port {}".format(*server_address))
    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)

    chosen_arm = None

    if means is not None:
        max_mean = np.max(means)
        max_regret_by_logt = float('-inf')
        max_estregret_by_logt = float('-inf')

    try:
        while True:
            # Wait for a connection
            print("Waiting for a connection...")
            connection, client_address = sock.accept()
            try:
                print("(New) connection from", client_address)

                # Receive the data in small chunks and react to it
                while True:
                    print("Learning algorithm = {} and chosen_arm = {}, at time t = {}:".format(policy, chosen_arm, policy.t))
                    print("\n  Its pulls   = {}...\n  Its rewards = {}...\n  ==> means   = {}...".format(policy.pulls, policy.rewards, policy.rewards / (1 + policy.pulls)))
                    if has_index:
                        print("  And internal indexes =", policy.index)

                    # print regret
                    if means is not None and policy.t > 1:
                        cumulated_rewards = np.sum(policy.rewards)
                        instant_regret = (max_mean * policy.t) - cumulated_rewards
                        instant_regret_by_logt = instant_regret / np.log(policy.t)
                        max_regret_by_logt = max(max_regret_by_logt, instant_regret_by_logt)
                        print("\n- Current instantaneous regret at time t = {} is = {:.3g}, and regret / log(t) = {:.3g}\n  and total max regret / log(t) already seen = {:.3g}...".format(policy.t, instant_regret, instant_regret_by_logt, max_regret_by_logt))  # DEBUG

                        estimated_rewards = np.sum(np.dot(means, policy.pulls))
                        estimated_regret = (max_mean * policy.t) - estimated_rewards
                        instant_estregret_by_logt = estimated_regret / np.log(policy.t)
                        max_estregret_by_logt = max(max_estregret_by_logt, instant_estregret_by_logt)
                        print("- Current estimated regret at time t = {} is = {:.3g}, and estimated regret / log(t) = {:.3g}.\n  and total max estimated regret / log(t) already seen = {:.3g} (based on pulls and means, not actual rewards)...".format(policy.t, estimated_regret, instant_estregret_by_logt, max_estregret_by_logt))  # DEBUG

                    data = connection.recv(16)
                    message = data.decode()
                    print("\nData received: {!r}".format(message))
                    try:
                        reward = float(message)

                        if chosen_arm is not None:
                            print("Passing reward {} on arm {} to the policy...".format(reward, chosen_arm))
                            policy.getReward(chosen_arm, reward)
                    except ValueError:
                        print("Unable to convert message = '{!r}' to a float reward...".format(message))  # DEBUG
                    try:
                        chosen_arm = policy.choice()
                    except ValueError:
                        chosen_arm = (policy.t + 1) % policy.nbArms
                        print("Unable to use policy's choice() method... playing the (t+1)%K-th = {} arm...".format(chosen_arm))  # DEBUG
                    message = str(chosen_arm)
                    print("Sending: '{!r}'...".format(message))
                    connection.sendall(message.encode())

            except ConnectionResetError:
                print("Remote connection was not found... waiting for the next one!")
            finally:
                # Clean up the connection
                print("Closing connection...")
                connection.close()
    finally:
        # Clean up the socket
        print("Closing socket...")
        sock.close()


def transform_str(params):
    """Like a safe :func:`exec()` on a dictionary that can contain special values:

    - strings are interpreted as variables names (e.g., policy names) from the current ``globals()`` scope,
    - list are transformed to tuples to be constant and hashable,
    - dictionary are recursively transformed.

    .. warning:: It is still as unsafe as :func:`exec` : only use it with trusted inputs!
    """
    for (key, value) in params.items():
        try:
            if isinstance(value, dict):
                transform_str(value)
            elif value in globals():
                params[key] = globals()[value]
        except TypeError:
            pass


def main(args):
    """
    Take args, construct the learning policy and starts the server.
    """
    host = str(args['--host'])
    port = int(args['--port'])
    try:
        means = str(args['--means'])
        means = means.replace('[', '').replace(']', '')
        means = [ float(m) for m in means.split(',') ]
        means = np.asarray(means, dtype=float)
    except ValueError:
        means = None

    json_configuration = args['<json_configuration>']
    configuration = read_configuration_policy(json_configuration)

    nbArms = int(configuration['nbArms'])
    # try to map strings in the dictionary to variables, e.g., policies
    params = configuration['params']
    transform_str(params)

    print("Params =", params)
    policy = globals()[configuration['archtype']](nbArms, **params)
    print("Using the policy", policy)

    return server(policy, host, port, means=means)


if __name__ == '__main__':
    arguments = docopt(__doc__, version=version)
    # print("arguments =", arguments)  # DEBUG
    main(arguments)
