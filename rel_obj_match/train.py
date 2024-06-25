"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

import os
import argparse
import ntpath
from datetime import datetime
from shutil import copyfile
import socket

# Disable TF info messages (needs to be done before tf import)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from utils.helper_funcs import *

RND_SEED = 42
set_seed(RND_SEED)

from utils.tf_funcs import *
from utils import tb_funcs
from utils import tools_eval


# GPU should not be allocated entirely at beginning
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
print(len(gpu_devices), gpu_devices)
if len(gpu_devices) > 0:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="hypersim", help="Data Set [default: hypersim]"
    )
    parser.add_argument(
        "--data_dir",
        default="/is/rg/ev/scratch/celich/data/Hypersim/Hypersim_Processed_ICRA23",
        help="Data dir [default: /is/rg/ev/scratch/celich/data/Hypersim/Hypersim_Processed_ICRA23]",
    )
    parser.add_argument(
        "--log_dir",
        default="experiments/log",
        help="Log dir [default: experiments/log]",
    )
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
    parser.add_argument(
        "--config", default="", help="[Optional] Specify specific config file."
    )
    return parser.parse_args()


# --------------------------------------------------
# --- Main Training


def main(model, cnfg, datasets):
    time_start = datetime.now()
    time_start_str = "{:%d.%m_%H:%M}".format(datetime.now())

    # Create dataset iterator for training and validation
    _, iterator_train = get_data_iterator(datasets["train"], train=True)
    _, iterator_val = get_data_iterator(datasets["val"])
    iterators = {"train": iterator_train, "val": iterator_val}

    print("+++++++++++++++++++++++++++++++")

    # Training parameters
    params = {}
    weights = cnfg["model"]["l-weights"]
    for k, w in weights.items():
        params["w-" + k] = w
    obj_count = datasets["train"].lbl_count
    params["obj_lbl_weight"] = np.divide(
        np.ones_like(obj_count, dtype=np.float32),
        np.asarray(obj_count, dtype=np.float32),
        out=np.zeros_like(obj_count, dtype=np.float32),
        where=obj_count != 0,
    )

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=cnfg["training"]["learning_rate"])

    # Operator to save and restore all the variables.
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=model)
    manager = tf.train.CheckpointManager(
        ckpt, os.path.join(LOG_DIR, "ckpts"), max_to_keep=10
    )

    # Writer for summary
    writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, "summary"))

    # Init variables
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        start_epoch = int(ckpt.step)
        print(
            "Restored from {}, epoch {}".format(manager.latest_checkpoint, start_epoch)
        )
    else:
        start_epoch = 1
        print("Start new training.")

    # Create ops dictionary
    ops = {"params": params, "iterators": iterators, "optimizer": opt}

    # Iterative Training
    max_epoch = cnfg["training"]["max_epoch"]

    for epoch in range(start_epoch, max_epoch + 1):
        LOG_FILE.write("----")
        time_now_str = "{:%d.%m_%H:%M}".format(datetime.now())
        LOG_FILE.write(
            "**** EPOCH %03d - start: %s - now: %s ****"
            % (epoch, time_start_str, time_now_str)
        )

        run_one_epoch(epoch, model, cnfg, datasets, ops, True, writer)

        if int(ckpt.step) % cnfg["training"]["save_epoch"] == 0 or epoch == max_epoch:
            save_path = manager.save()
            LOG_FILE.write(
                "Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path)
            )
            run_one_epoch(epoch, model, cnfg, datasets, ops, False, writer)
        ckpt.step.assign_add(1)

        # For cluster: break training after some time
        time_delta = datetime.now() - time_start
        time_delta_hours, time_delta_minutes, time_delta_seconds = split_time_delta(
            time_delta
        )
        print(
            "Runtime: {} day(s), {} hour(s), {} minute(s), {} seconds".format(
                time_delta.days,
                time_delta_hours,
                time_delta_minutes,
                time_delta_seconds,
            )
        )


def run_one_epoch(epoch, net, cnfg, datasets, ops, is_training, summary_writer=None):
    LOG_FILE.write("----")

    if is_training:
        split = "train"
    else:
        split = "val"
        LOG_FILE.write("EVAL")
    data = datasets[split]
    iterator = ops["iterators"][split]

    num_batches = int(np.ceil((data.get_size() / data.bs)))
    b_mod = 10 if (num_batches < 150) else 250

    loss_dict = {}
    eval_dict = {"general_scene-name_info": data.scene_names}

    for batch_id in range(num_batches):

        if batch_id % b_mod == 0:
            print("Current batch/total batch num: %d/%d" % (batch_id, num_batches))

        input_batch = next(iterator)
        input_batch = net.get_input(input_batch)

        if is_training:
            output, losses = train_step(
                net, input_batch, ops["optimizer"], ops["params"]
            )
        else:  # validation
            output, losses, eval_dict = test_step(
                net, input_batch, eval_dict, cnfg, params=ops["params"]
            )

        for k, v in losses.items():
            if k[:1] == "l":
                if k not in loss_dict:
                    loss_dict[k] = 0.0
                loss_dict[k] += v / num_batches

    if summary_writer is not None:
        summary_output_dir = os.path.join(LOG_DIR, split, f"ep{epoch}_xxx.jpg")
        summary_data = net.get_summary_data(
            input_batch,
            output,
            loss_dict,
            eval_dict,
            cnfg,
            is_training,
            output_path=summary_output_dir,
        )
        for k, v in ops["params"].items():
            if k[0:2] == "w-":  # loss weight
                summary_data["params"][k] = v
        if not is_training:
            eval_dict = tools_eval.finalize_eval_scores(
                eval_dict, print_out=True, log_file=LOG_FILE
            )
            summary_data["eval"] = eval_dict
        tb_funcs.summarize_all(
            summary_writer,
            epoch,
            summary_data,
            imgs_types=[],
            log_file=LOG_FILE,
            mode=split,
        )

    if is_training and np.isnan(loss_dict["loss_total"]):
        LOG_FILE.write("[ERROR] NaN error! Break Training.")
        exit(1)


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

    if FLAGS.config != "":
        cnfg_name = FLAGS.config
    else:
        cnfg_name = "cnfg_{}_{}".format(model_base_name, FLAGS.dataset)

    # - Continue training
    if os.path.exists(os.path.join(LOG_DIR, "ckpts", "checkpoint")):
        model_file = os.path.join(LOG_DIR, model_base_name + ".py")
        model_module = load_module_from_log(model_base_name, model_file)

        data_provider_file = os.path.join(LOG_DIR, "data_provider.py")
        data_provider_module = load_module_from_log("data_provider", data_provider_file)

        cnfg_file = os.path.join(LOG_DIR, cnfg_name + ".py")
        cnfg_module = load_module_from_log(cnfg_name, cnfg_file)
        cnfg = cnfg_module.cnfg_dict

        new_training = False
    # - Start new training
    else:
        # Create all log dirs if they do not exist
        log_dirs = [FLAGS.log_dir, LOG_DIR]
        for log_dir in log_dirs:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
                os.mkdir(os.path.join(log_dir, "train"))
                os.mkdir(os.path.join(log_dir, "val"))

        model_file = os.path.join("models", model_base_name + ".py")
        model_module = importlib.import_module("." + model_base_name, "models")

        data_provider_file = os.path.join("utils", "data_provider.py")
        data_provider_module = importlib.import_module(".data_provider", "utils")

        cnfg, cnfg_file = load_cnfg(cnfg_name, base_dir="config")

        # Back up files for later inspection
        proj_dir = os.path.dirname(os.path.abspath(__file__))
        backup_files = [
            model_file,
            cnfg_file,
            "train.py",
            data_provider_file,
            os.path.join("models", "losses.py"),
            os.path.join("models", "networks.py"),
        ]
        for backup_file in backup_files:
            backup_path = os.path.join(LOG_DIR, ntpath.split(backup_file)[-1])
            if os.path.exists(backup_path):
                os.remove(backup_path)
            copyfile(os.path.join(proj_dir, backup_file), backup_path)

        new_training = True

    model = model_module.get_model(cnfg, model_ext_name)

    # Load data
    print("Load data..")
    datasets = {
        "train": data_provider_module.get_dataset(
            FLAGS.data_dir, model_ext_name, "train", cnfg["data"]
        ),
        "val": data_provider_module.get_dataset(
            FLAGS.data_dir, model_ext_name, "val", cnfg["data"]
        ),
    }
    datasets["train"].set_bs(cnfg["training"]["batch_size"])

    # Open Log-file
    LOG_FILE = LogFile(os.path.join(LOG_DIR, "log_train.txt"))
    LOG_FILE.write(str(FLAGS))
    LOG_FILE.write("{:%d.%m_%H:%M}".format(datetime.now()))

    main(model, cnfg, datasets)

    exit(0)
