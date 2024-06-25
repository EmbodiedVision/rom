"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

import os
import argparse
from datetime import datetime
import json
import pickle

# Disable TF info messages (needs to be done before tf import)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from utils.helper_funcs import *
from utils.tf_funcs import *
from utils import tools_eval


# GPU should not be allocated entirely at beginning
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if len(gpu_devices) > 0:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)


LEVELS = [[0], [1], [2], [0, 1, 2]]  # allows easy parallel evaluation on cluster


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="hypersim", help="Data Set [default: 3rscan, (...)]"
    )
    parser.add_argument(
        "--data_dir",
        default="./../datasets/3RScan/data/Hypersim_ProcessedV2-allCams-allSceneType",
        help="Data dir [default: ./../Hypersim_ProcessedV2-allCams-allSceneType]",
    )
    parser.add_argument("--log_dir", default="log", help="Log dir [default: log]")
    parser.add_argument("--eval_dir", default="eval", help="Log dir [default: eval]")
    parser.add_argument("--split", default="test", help="Log dir [default: test, val]")
    parser.add_argument("--lvl", default=-1, type=int, help="Log dir [default: -1]")
    parser.add_argument(
        "--model",
        default="romnet-matchVflex",
        help="Model name [e.g. romnet-matchVflex]",
    )
    parser.add_argument(
        "--message",
        default="",
        help="Message that specifies settings etc. (for log dir name)",
    )
    return parser.parse_args()


# --------------------------------------------------
# --- Main Evaluation


def main(model, cnfg, data, eval_dir):
    time_start_str = "{:%d.%m_%H:%M}".format(datetime.now())

    # Create dataset iterator for testing
    _, iterator_test = get_data_iterator(data)

    print("+++++++++++++++++++++++++++++++")

    # Operator to save and restore all the variables.
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
    manager = tf.train.CheckpointManager(
        ckpt, os.path.join(LOG_DIR, "ckpts"), max_to_keep=10
    )

    # Init variables
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        start_epoch = int(ckpt.step)
        print(
            "Restored from {}, epoch {}".format(manager.latest_checkpoint, start_epoch)
        )
    else:
        print("No pre-trained model found.")
        exit(1)
    eval_sub_dir = (
        "ep"
        + str(start_epoch)
        + "_"
        + FLAGS.split
        + "_lvl-"
        + "-".join(str(l) for l in cnfg["data"]["lvl_difficulty"][FLAGS.split])
    )
    eval_dir = os.path.join(os.path.join(eval_dir, eval_sub_dir))
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
        os.mkdir(os.path.join(eval_dir, "imgs"))

    # Open Log-file (do here to have separate log file per evaluation)
    global LOG_FILE
    LOG_FILE = LogFile(os.path.join(eval_dir, "log_eval.txt"))
    LOG_FILE.write(str(FLAGS))
    LOG_FILE.write("{:%d.%m_%H:%M}".format(datetime.now()))
    LOG_FILE.write(
        "Restored from {}, epoch {}".format(manager.latest_checkpoint, start_epoch)
    )

    # Create ops dictionary
    ops = {
        "iterator": iterator_test,
    }

    LOG_FILE.write("----")
    LOG_FILE.write("**** EVAL- start: %s ****" % (time_start_str))

    eval_model(model, cnfg, data, ops, eval_dir)


def eval_model(net, cnfg, data, ops, eval_dir):

    num_batches = int(data.get_size())
    b_mod = 100 if (num_batches < 1500) else 1000

    eval_dict = {"general_scene-name_info": data.scene_names}

    for batch_id in range(num_batches):

        if batch_id % b_mod == 0:
            print("Current image/total image num: %d/%d" % (batch_id, num_batches))

        input_batch = next(ops["iterator"])
        input_batch = net.get_input(input_batch)

        output = net(input_batch)

        data = {**input_batch, **output}
        for k in data.keys():
            if (
                tf.is_tensor(data[k])
                and ("obj_" in k or "rel_" in k)
                and not ("_pred" in k)
            ):
                data[k] = tf.expand_dims(data[k], axis=0)
            else:
                data[k] = data[k]

        tools_eval.eval_quantitative_single_batch_class(
            data, eval_dict, ["obj_cls_gt", "obj_cls_pred"], cnfg
        )
        tools_eval.eval_quantitative_single_batch_pose(data, eval_dict, cnfg)
        tools_eval.eval_quantitative_single_batch_rel(data, eval_dict)

        if batch_id % b_mod == 0 and batch_id > 0:
            print("[tmp]")
            _ = tools_eval.finalize_eval_scores(eval_dict, print_out=True)

    print("[final]")
    time_end_str = "{:%d.%m_%H:%M}".format(datetime.now())
    LOG_FILE.write("----")
    LOG_FILE.write("**** EVAL- done: %s ****" % (time_end_str))

    eval_dict_updated = tools_eval.finalize_eval_scores(eval_dict)

    tools_eval.print_eval_scores(eval_dict_updated, log_file=LOG_FILE)

    with open(os.path.join(eval_dir, f"eval_props.pickle"), "wb") as handle:
        pickle.dump(eval_dict_updated, handle, protocol=pickle.HIGHEST_PROTOCOL)

    eval_dict_sub = {
        k: float(eval_dict_updated[k])
        for k in eval_dict_updated.keys()
        if k[:5] == "score"
    }
    with open(os.path.join(eval_dir, f"eval_props.json"), "w") as f:
        json.dump(eval_dict_sub, f, indent="")


# --------------------------------------------------
# ---


if __name__ == "__main__":

    FLAGS = parse_arguments()

    if FLAGS.message == "":
        print("Need to specify message/ name for  model [--message]")
        exit(0)
    model_base_name, model_ext_name = FLAGS.model.split("-")

    # Load model and config file
    print("Load model and config file..")
    LOG_DIR = os.path.join(
        FLAGS.log_dir, FLAGS.dataset + "_" + FLAGS.model + "(" + FLAGS.message + ")"
    )
    cnfg_name = "cnfg_{}_{}".format(model_base_name, FLAGS.dataset)
    # - Look for pre-trained model
    if os.path.exists(os.path.join(LOG_DIR, "ckpts", "checkpoint")):
        # model_file = os.path.join(LOG_DIR, model_base_name + '.py')
        # model_module = load_module_from_log(model_base_name, model_file)

        # data_provider_file = os.path.join(LOG_DIR, 'data_provider.py')
        # data_provider_module = load_module_from_log('data_provider', data_provider_file)

        # cnfg_file = os.path.join(LOG_DIR, cnfg_name + '.py')
        # cnfg_module = load_module_from_log(cnfg_name, cnfg_file)
        # cnfg = cnfg_module.cnfg_dict

        # # -> use current version
        model_file = os.path.join("models", model_base_name + ".py")
        model_module = importlib.import_module("." + model_base_name, "models")
        data_provider_module = importlib.import_module(".data_provider", "utils")
        cnfg_name = "cnfg_{}_{}".format(model_base_name, FLAGS.dataset)
        cnfg, cnfg_file = load_cnfg(cnfg_name, base_dir="config")

        new_training = False
    else:
        print("No model found at {}".format(LOG_DIR))
        exit(0)

    eval_dir = os.path.join(os.path.join(LOG_DIR, FLAGS.eval_dir))
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    model = model_module.get_model(cnfg, model_ext_name)

    # Load data
    print("Load data..")
    if not (
        -1 <= FLAGS.lvl < len(LEVELS)
    ):  # -1 is default, indicates evaluation over all levels jointly
        print("Please enter valid level [--lvl]")
        exit(0)
    cnfg["data"]["lvl_difficulty"][FLAGS.split] = LEVELS[FLAGS.lvl]
    dataset = data_provider_module.get_dataset(
        FLAGS.data_dir, model_ext_name, FLAGS.split, cnfg["data"]
    )

    LOG_FILE = None

    main(model, cnfg, dataset, eval_dir)

    exit(0)
