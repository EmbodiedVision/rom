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
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from utils.helper_funcs import *
from utils.tf_funcs import *
from utils import tools_eval


# GPU should not be allocated entirely at beginning
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if len(gpu_devices) > 0:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)


LEVELS = [[0], [1], [2], [0, 1, 2]]  # allows easy parallel evaluation on cluster

COMBI_MODI = ["ROM-feature", "Object-level_Keypoint-based_Matching", "Combination"]
# COMBI_MODI = ['ROM-feature']


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="hypersim", help="Data Set [default: hypersim, (...)]"
    )
    parser.add_argument(
        "--data_dir",
        default="/is/rg/ev/scratch/celich/data/Hypersim/Hypersim_Processed_ICRA23",
        help="Data dir [default: .../Hypersim_Processed_ICRA23]",
    )
    parser.add_argument(
        "--keypoint_matching_res_dir",
        default="/is/rg/ev/scratch/celich/data/Hypersim/Hypersim_SuperGlue-keypoints_ICRA23/processed_results",
        help="Preprocessed results of Keypoint-based matching [optional]",
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
        exit(0)
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

    eval_model(model, data, ops, eval_dir)


def eval_model(net, data, ops, eval_dir):

    num_batches = int(data.get_size())
    b_mod = 100 if (num_batches < 1500) else 1000

    eval_dict_combi = {}
    # eval_dict_combi_time = {}
    imgs_dirs = {}
    for cm in COMBI_MODI:
        eval_dict_combi[cm] = {"general_scene-name_info": data.scene_names}
        # eval_dict_combi_time[cm] = {}
        imgs_dirs[cm] = os.path.join(os.path.join(eval_dir, "imgs"), cm)
        if not os.path.exists(imgs_dirs[cm]):
            os.mkdir(imgs_dirs[cm])
    best_f1score = 0
    best_res_wrt_f1score = {}

    for batch_id in range(num_batches):

        if batch_id % b_mod == 0:
            print("Current image/total image num: %d/%d" % (batch_id, num_batches))

        input_batch = next(ops["iterator"])
        input_batch = net.get_input(input_batch)

        names = data.get_sample_names(input_batch)
        scene_names = names["scene_names"]
        cam_names = names["cam_names"]
        frame_names = names["frame_names"]
        # if scene_names[0] != 'ai_003_010':  # use e.g. to quickly produce visualization for specific scene
        #     continue

        # time_rom_net_start = datetime.now()
        output = net(input_batch)

        sim_mat = output["obj_sim_mat"][0].numpy()
        n_obj = int(max(sim_mat.shape[0], sim_mat.shape[1]))

        # time_rom_net_end = datetime.now()
        # time_rom_net = time_rom_net_end - time_rom_net_start

        if (
            "Object-level_Keypoint-based_Matching" in COMBI_MODI
            or "Combination" in COMBI_MODI
        ):
            # get pre-computed superglue keypoint matching results
            # pair_name = f'{cam_names[0]}-{frame_names[0]}_{cam_names[1]}-{frame_names[1]}.npy'
            pair_name = "{}-{}_{}-{}".format(
                cam_names[0].replace("_", ""),
                frame_names[0].replace(".", ""),
                cam_names[1].replace("_", ""),
                frame_names[1].replace(".", ""),
            )
            vote_path = os.path.join(
                KEYPONT_MATCHING_RES_DIR,
                scene_names[0],
                "keypoint-votes_{}.npy".format(pair_name),
            )
            vote_mat = np.load(vote_path)

        for cm in COMBI_MODI:
            # time_match_start = datetime.now()
            if cm == "ROM-feature":
                wght_aff_mat = np.exp(sim_mat)
            elif "Keypoint" in cm:
                wght_aff_mat = vote_mat
            elif "Combination" in cm:
                wght_aff_mat = np.exp(sim_mat) + 100.0 * vote_mat
            else:
                print(f"[ERROR] eval_combi: Unknown modus {cm}.")

            log_wght_aff_mat = np.log(np.where(wght_aff_mat > 1e-8, wght_aff_mat, 1e-8))
            logopt_wght_aff_mat = tools_eval.sinkhorn(log_wght_aff_mat, alpha=1.0)

            output["obj_affinity_mat"] = np.expand_dims(logopt_wght_aff_mat, axis=0)
            # time_match_end = datetime.now()
            # time_match_full = time_match_end - time_match_start

            # if cm == 'ROM-feature' or cm == 'Combination':
            #     time_match_full += time_rom_net
            # if n_obj in eval_dict_combi_time[cm].keys():
            #     eval_dict_combi_time[cm][n_obj]['time'] += time_match_full
            #     eval_dict_combi_time[cm][n_obj]['occ'] += 1
            # else:
            #     eval_dict_combi_time[cm][n_obj] = {}
            #     eval_dict_combi_time[cm][n_obj]['time'] = time_match_full
            #     eval_dict_combi_time[cm][n_obj]['occ'] = 1

            # -> only evaluation on matching task
            tools_eval.eval_quantitative_single_batch_affinity(
                {**input_batch, **output},
                eval_dict_combi[cm],
                # img_dir=imgs_dirs[cm]  # outcomment in case that qualitative samples should be saved
            )

    eval_dict_combi_updated = {}
    for cm in COMBI_MODI:
        eval_dict_combi_updated[cm] = tools_eval.finalize_eval_scores(
            eval_dict_combi[cm]
        )
        if eval_dict_combi_updated[cm]["score_match_f1"] > best_f1score:
            best_f1score = eval_dict_combi_updated[cm]["score_match_f1"]
            best_res_wrt_f1score = eval_dict_combi_updated[cm]
            best_res_wrt_f1score["cm"] = cm
        # for n_obj in eval_dict_combi_time[cm].keys():
        #     eval_dict_combi_time[cm][n_obj]['time_avg'] = eval_dict_combi_time[cm][n_obj]['time'] / eval_dict_combi_time[cm][n_obj]['occ']

    time_end_str = "{:%d.%m_%H:%M}".format(datetime.now())
    LOG_FILE.write("****     - done: %s ****" % (time_end_str))

    for cm in COMBI_MODI:
        LOG_FILE.write(f"==> CM={cm}")
        tools_eval.print_eval_scores(eval_dict_combi_updated[cm], log_file=LOG_FILE)

        # LOG_FILE.write(f'time:')
        # for n_obj in eval_dict_combi_time[cm].keys():
        #     avg_time = eval_dict_combi_time[cm][n_obj]['time_avg']
        #     n_occ = eval_dict_combi_time[cm][n_obj]['occ']
        #     # avg_time_str = '{%S}'.format(avg_time)
        #     # print(f'\tn_obj={n_obj}: {avg_time_str} ({avg_time})')
        #     LOG_FILE.write(f'\tn_obj={n_obj} ({n_occ} occ.): ({avg_time})')

        prefix = f"cm-{cm}_"
        with open(os.path.join(eval_dir, prefix + "eval.pickle"), "wb") as handle:
            pickle.dump(
                eval_dict_combi_updated[cm], handle, protocol=pickle.HIGHEST_PROTOCOL
            )

        eval_dict_sub = {
            k: float(eval_dict_combi_updated[cm][k])
            for k in eval_dict_combi_updated[cm].keys()
            if k[:5] == "score"
        }
        with open(os.path.join(eval_dir, prefix + "eval.json"), "w") as f:
            json.dump(eval_dict_sub, f, indent="")

    cm = best_res_wrt_f1score["cm"]
    if len(COMBI_MODI) > 1:
        LOG_FILE.write(f"==> BEST: CM={cm} with f1={best_f1score:.3}")
        tools_eval.print_eval_scores(best_res_wrt_f1score, log_file=LOG_FILE)


# --------------------------------------------------
# ---

if __name__ == "__main__":

    FLAGS = parse_arguments()

    if FLAGS.message == "":
        print("Need to specify message/ name for  model [--message]")
        exit(0)
    model_base_name, model_ext_name = FLAGS.model.split("-")

    # Load model and config file
    LOG_DIR = os.path.join(
        FLAGS.log_dir, FLAGS.dataset + "_" + FLAGS.model + "(" + FLAGS.message + ")"
    )
    cnfg_name = "cnfg_{}_{}".format(model_base_name, FLAGS.dataset)
    # - Look for pre-trained model
    if os.path.exists(os.path.join(LOG_DIR, "ckpts", "checkpoint")):
        # # -> use code copy from training
        # model_file = os.path.join(LOG_DIR, model_base_name + '.py')
        # model_module = load_module_from_log(model_base_name, model_file)

        # data_provider_file = os.path.join(LOG_DIR, 'data_provider.py')
        # data_provider_module = load_module_from_log('data_provider', data_provider_file)

        # cnfg_file = os.path.join(LOG_DIR, cnfg_name + '.py')
        # cnfg_module = load_module_from_log(cnfg_name, cnfg_file)
        # cnfg = cnfg_module.cnfg_dict

        # new_training = False

        # -> use current version
        model_file = os.path.join("models", model_base_name + ".py")
        model_module = importlib.import_module("." + model_base_name, "models")
        data_provider_module = importlib.import_module(".data_provider", "utils")
        cnfg_name = "cnfg_{}_{}".format(model_base_name, FLAGS.dataset)
        cnfg, cnfg_file = load_cnfg(cnfg_name, base_dir="config")
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
    KEYPONT_MATCHING_RES_DIR = os.path.join(
        FLAGS.keypoint_matching_res_dir, FLAGS.split
    )

    LOG_FILE = None

    main(model, cnfg, dataset, eval_dir)

    exit(0)
