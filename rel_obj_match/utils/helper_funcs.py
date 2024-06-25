"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

import os
import importlib
import numpy as np
from termcolor import colored


# --------------------------------------------------
# --- General Functions


def set_seed(rnd_seed):
    from random import seed

    seed(rnd_seed)
    from numpy.random import seed

    seed(rnd_seed)
    from tensorflow import random as random_tf

    random_tf.set_seed(rnd_seed)


def check_if_dir_exists(directory_path):
    if not os.path.exists(directory_path):
        print(
            colored(
                "(ERROR) Input directory does not exist - {}".format(directory_path),
                "red",
            )
        )
        for _ in range(4):
            directory_path = os.path.dirname(directory_path)
            print(directory_path, os.path.exists(directory_path))
        exit(0)


def load_module_from_log(name, file):
    """
    Load module from backup file (i.e. not in project structure)
    :param name:    string, name of module
    :param file:    string, path to module file
    :return:        imported module
    """
    assert name in file
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_cnfg(name, base_dir="config"):

    cnfg_file = os.path.join(base_dir, name + ".py")
    config_module = importlib.import_module("." + name, "config")
    cnfg = config_module.cnfg_dict
    return cnfg, cnfg_file


def split_time_delta(time_delta):
    time_delta_hours = time_delta.seconds // 3600
    time_delta_minutes = (time_delta.seconds - time_delta_hours * 3600) // 60
    time_delta_seconds = (
        time_delta.seconds - time_delta_hours * 3600 - time_delta_minutes * 60
    )

    return [time_delta_hours, time_delta_minutes, time_delta_seconds]


class LogFile:

    def __init__(self, log_path, new=False):
        self.log_path = log_path
        if new:
            self.log_fout = open(log_path, "w")
        else:
            self.log_fout = open(log_path, "a+")

    def write(self, out_str):
        self.log_fout.write(out_str + "\n")
        self.log_fout.flush()
        print(out_str)

    def read_lines(self):
        fp = open(self.log_path, "r")
        lines = fp.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace("\n", "").split(" ")
        return lines


# --------------------------------------------------
# ---  Mapping


def mapping_with_dict(img, out_shape, transfer_dict):
    res = np.zeros(out_shape)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    for k, v in transfer_dict.items():
        res += np.where(img == k, v, np.zeros_like(v))
    return res
