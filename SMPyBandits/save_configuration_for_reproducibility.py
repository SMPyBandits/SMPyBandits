#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Small utility to save the configuration dictionnary and file."""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import os.path
import shutil
import re
import pprint

def save_configuration_for_reproducibility(
        configuration=dict(),
        configuration_module=None,
        plot_dir="plots/",
        hashvalue=0,
        main_name="main.py",
    ):
    """ Save configuration_*.py FILE and configuration_*.configuration DICTIONNARY in the plot_dir with a certain hashvalue.

    - See https://github.com/SMPyBandits/SMPyBandits/issues/179 for more details.
    """
    if os.path.exists(configuration_module.__file__):
        configuration_filename = configuration_module.__file__
        file_start = configuration_filename
        file_end = os.path.join(plot_dir, os.path.basename(configuration_filename).replace('.py', '__{}.py'.format(hashvalue)))
        print("Copying {} to {}...".format(file_start, file_end))  # DEBUG
        try:
            shutil.copyfile(file_start, file_end)
        except FileNotFoundError:
            print("WARNING could not save file {} to {}, maybe you are not in the correct folder?\nSkipping this step...".format(file_start, file_end))  # DEBUG

    # --- DONE Save just the configuration to a minimalist python file
    file_end_just_dict = file_end.replace('.py', '_minimalist.py')
    print("Saving full configuration dictionnary to {}...".format(file_end_just_dict))  # DEBUG

    with open(file_end_just_dict, 'w') as f:
        str_configuration = pprint.pformat(configuration).\
            replace("<class '", "SMPyBandits.").replace("'>", "").\
            replace("<built-in function ", "").replace(">", "")
            # FIXME other things to do!

        relative_imports_to_do = []
        absolute_imports_to_do = []
        for pattern in sorted(list(set(re.findall('SMPyBandits.[A-Za-z0-9_.]*', str_configuration)))):
            module, classname = '.'.join(pattern.split('.')[:-1]).replace('SMPyBandits.', ''), pattern.split('.')[-1]
            relative_imports_to_do.append("from {} import {}".format(module, classname))
            absolute_imports_to_do.append("from SMPyBandits.{} import {}".format(module, classname))
            str_configuration = str_configuration.replace(pattern, classname)

        f.write("""# -*- coding: utf-8 -*-
\"\"\"Minimalist file to reproduce experiments with hash = {}.\"\"\"
from __future__ import division, print_function  # Python 2 compatibility
__author__ = "Lilian Besson"
try:
    from Arms import *
    from Policies import *
    from Policies.kullback import *
    from PoliciesMultiPlayers import *
    {}
except ImportError:
    from SMPyBandits.Arms import *
    from SMPyBandits.Policies import *
    from SMPyBandits.Policies.kullback import *
    from SMPyBandits.PoliciesMultiPlayers import *
    {}

configuration = {}

# use it with:
# $ python {} {}
""".format(hashvalue,
            "\n    ".join(relative_imports_to_do),
            "\n    ".join(absolute_imports_to_do),
            str_configuration,
            main_name,
            file_end_just_dict,
        )
    )
    # TODO do the same on other main_*.py scripts
