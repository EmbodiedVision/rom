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

import numpy as np
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from utils import data_provider
from utils.tf_funcs import get_data_iterator
from utils.helper_funcs import load_cnfg
from models.romnet import get_model
from utils import viz


LEVELS = [[0], [1], [2], [0, 1, 2]]  # allows easy parallel evaluation on cluster


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/is/rg/ev/scratch/celich/data/Hypersim/Hypersim_Processed_ICRA23",
    )
    parser.add_argument(
        "--keypoint_matching_res_dir",
        default="/is/rg/ev/scratch/celich/data/Hypersim/Hypersim_SuperGlue-keypoints_ICRA23",
    )
    parser.add_argument(
        "--output_dir",
        default="/is/rg/ev/scratch/celich/data/Hypersim/Hypersim_SuperGlue-keypoints_ICRA23/processed_results",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--lvl", default=-1, type=int, help="Log dir [default: -1]")
    parser.add_argument("--detector", default="")
    parser.add_argument(
        "--run_on_cluster",
        type=int,
        choices=(0, 1),
        default=0,
        help="indicator whether job is run on cluster (for simple parallelization) [0 or 1; default=0]",
    )
    return parser.parse_args()


# --------------------------------------------------
# --- Helper Function


def get_input_with_detections(batch_data):
    return {
        "scan_ids": batch_data[0][0],  # (BS=1*2, 1)  --
        "cam_ids": batch_data[1][0],  # (BS*2, 1) --
        "frame_ids": batch_data[2][0],  # (BS*2, 1) --
        "scene_rgb": batch_data[3][0],  # (BS*2, W, H, 3)
        "cam_extr": batch_data[4][0],  # (BS*2, 4, 4)
        "mpa": batch_data[5][0],  # (BS*2,)
        "scene_split": batch_data[6][0],  # (BS*2)
        "obj_ids": batch_data[7][0],  # (BSxN_obj)
        "obj_ids_org": batch_data[8][0],  # (BSxN_obj')
        "obj_rgb": batch_data[9][0],  # (BSxN_obj, W', H', 3)
        "obj_bb2d": batch_data[10][0],  # (BSxN_obj, 4)
        "obj_cls_gt": batch_data[11][0],  # (BSxN_obj, K_obj)
        "obj_frame_id": batch_data[12][0],  # (BSxN_obj,)
        "obj_split": batch_data[13][0],  # (BSxN_obj,)
        "rel_ids": batch_data[14][0],  # (BSxN_rel, 2)
        "rel_frame_id": batch_data[15][0],  # (BSxN_rel,)
        "rel_split": batch_data[16][0],  # (BSxN_rel,)
    }


# --------------------------------------------------
# --- Main Evaluation


def get_vote_matrix_for_keypoint_matches(
    match_path, input_batch, image_shape, debug_viz=False
):

    h, w = image_shape
    matches = np.load(match_path)  # (2, N_matches, 2)
    matches = np.stack([matches[:, :, 1], matches[:, :, 0]], axis=-1)

    # -- viz
    if debug_viz:
        scene_rgb = input_batch["scene_rgb"].numpy()
        scene_rgb_match = viz.draw_keypoint_matches(scene_rgb, matches)
        viz.show_img_grid([[scene_rgb_match]], [["rgb"]])
    # ---

    obj_frame_id = input_batch["obj_frame_id"]
    obj_lbl_onehot = input_batch["obj_cls_gt"]
    fid_obj = []
    obj_bb2d = []
    obj_bb_size = []
    obj_cls = []
    for fid in [0, 1]:
        fid_obj.append((obj_frame_id == fid).numpy())
        obj_bb2d.append(input_batch["obj_bb2d"][fid_obj[fid]])
        obj_bb_size.append(
            (obj_bb2d[fid][:, 2] - obj_bb2d[fid][:, 0])
            * (obj_bb2d[fid][:, 3] - obj_bb2d[fid][:, 1])
        )
        obj_cls.append(np.argmax(obj_lbl_onehot[fid_obj[fid]], axis=1))

    vote_obj_matches = np.zeros((obj_bb2d[0].shape[0], obj_bb2d[1].shape[0]))
    n_k = 0
    matches_new = []
    for m_idx in range(matches.shape[1]):
        match = matches[:, m_idx]
        match = [[match[i][0] / h, match[i][1] / w] for i in [0, 1]]
        for i in range(obj_bb2d[0].shape[0]):
            bb1 = obj_bb2d[0][i]
            if (
                (match[0][0] >= bb1[0])
                and (match[0][0] <= bb1[2])
                and (match[0][1] >= bb1[1])
                and (match[0][1] <= bb1[3])
            ):
                for j in range(obj_bb2d[1].shape[0]):
                    bb2 = obj_bb2d[1][j]
                    if (
                        (match[1][0] >= bb2[0])
                        and (match[1][0] <= bb2[2])
                        and (match[1][1] >= bb2[1])
                        and (match[1][1] <= bb2[3])
                    ):
                        vote_obj_matches[i, j] += 1
                        n_k += 1
                        matches_new.append(match)

    # -- viz
    if debug_viz:
        scene_rgb_match = []
        for fid in [0, 1]:
            scene_rgb_match.append(
                viz.draw_bb(scene_rgb[fid], obj_bb2d[fid], sem_lbls=obj_cls[fid])
            )
        # viz.show_img_grid([scene_rgb_match], [['rgb', 'rgb']])
        matches_new = np.stack(matches_new, axis=1)
        matches_new = np.asarray([[[h, w]]]) * matches_new
        scene_rgb_match = viz.draw_keypoint_matches(scene_rgb_match, matches_new)
        viz.show_img_grid([[scene_rgb_match]], [["rgb"]])
    # ---

    vote_obj_matches = vote_obj_matches / (
        np.expand_dims(obj_bb_size[0], axis=1) + np.expand_dims(obj_bb_size[1], axis=0)
    )

    return vote_obj_matches


def main(
    data, split, keypoint_matching_res_dir, eval_output_dir, cnfg, use_detections=False
):
    time_start_str = "{:%d.%m_%H:%M}".format(datetime.now())

    _, iterator = get_data_iterator(data)
    ops = {
        "iterator": iterator,
    }

    print("+++++++++++++++++++++++++++++++")
    print(
        "**** VARIANT: %s ****"
        % ("-".join([str(lvl) for lvl in cnfg["data"]["lvl_difficulty"][split]]))
    )
    print("**** EVAL- start: %s ****" % (time_start_str))
    print("----")

    num_batches = int(data.get_size())
    b_mod = 100 if (num_batches < 1500) else 1000

    eval_dict = {"general_scene-name_info": data.scene_names}

    seen_scenes = []

    batch_id_list = list(range(num_batches))
    if RUN_ON_CLUSTER:
        random.shuffle(batch_id_list)
    for batch_id in batch_id_list:  # evaluation on cluster

        if batch_id % b_mod == 0:
            print("Current image/total image num: %d/%d" % (batch_id, num_batches))

        input_batch = next(ops["iterator"])
        if use_detections:
            input_batch = get_input_with_detections(input_batch)
        else:
            input_batch = ROM_MODEL.get_input(input_batch)

        scan_id = input_batch["scan_ids"][0].numpy()
        cam_ids = input_batch["cam_ids"].numpy()
        frame_ids = input_batch["frame_ids"].numpy()
        room_name = eval_dict["general_scene-name_info"][scan_id]

        if room_name not in seen_scenes:
            print(f"  -- evaluate scene {room_name}")
            seen_scenes.append(room_name)

        cam_names = [data.cam_names[scan_id][cam_ids[i]] for i in [0, 1]]
        frame_names = [
            data.rgb_frame_names[scan_id][cam_ids[i]][frame_ids[i]] for i in [0, 1]
        ]
        pair_name = "{}-{}_{}-{}".format(
            cam_names[0].replace("_", ""),
            frame_names[0].replace(".", ""),
            cam_names[1].replace("_", ""),
            frame_names[1].replace(".", ""),
        )

        vote_name = "keypoint-votes_{}.npy".format(pair_name)
        vote_path = os.path.join(eval_output_dir, room_name, vote_name)
        if os.path.exists(vote_path):
            continue
        else:
            match_name = "matches_{}.npy".format(pair_name)
            match_path = os.path.join(keypoint_matching_res_dir, room_name, match_name)
            if not os.path.exists(match_path):
                print(match_path, " does not exist")
                continue

            if not os.path.exists(os.path.join(eval_output_dir, room_name)):
                os.mkdir(os.path.join(eval_output_dir, room_name))

            vote_obj_matches = get_vote_matrix_for_keypoint_matches(
                match_path, input_batch, data.img_shape_org, debug_viz=False
            )
            np.save(vote_path, vote_obj_matches)


# --------------------------------------------------


if __name__ == "__main__":

    FLAGS = parse_arguments()

    RUN_ON_CLUSTER = FLAGS.run_on_cluster

    if FLAGS.detector != "" and FLAGS.detector not in FLAGS.output_dir:
        FLAGS.output_dir += "_" + FLAGS.detector

    output_dir = os.path.join(FLAGS.output_dir, FLAGS.split)

    cnfg, cnfg_file = load_cnfg("cnfg_romnet_hypersim", base_dir="config")
    if not (-1 <= FLAGS.lvl < len(LEVELS)):  # -1 is default, indicates all levels
        print("Please enter valid level [--lvl]")
        exit(0)
    cnfg["data"]["lvl_difficulty"][FLAGS.split] = LEVELS[FLAGS.lvl]

    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    ROM_MODEL = get_model(cnfg, "matchVflex")

    v_name = "-".join([str(lvl) for lvl in cnfg["data"]["lvl_difficulty"][FLAGS.split]])
    print(str(FLAGS))
    print("{:%d.%m_%H:%M}".format(datetime.now()))

    # Load data
    print("Load data..")
    if FLAGS.detector == "":  # GT BB
        dataset = data_provider.get_dataset(
            FLAGS.data_dir, "matchVflex", FLAGS.split, cnfg["data"]
        )
    else:
        dataset = data_provider.get_dataset(
            FLAGS.data_dir, "matchVdetect", FLAGS.split, cnfg["data"], FLAGS.detector
        )
        dataset.set_model_type("matchVflex")

    main(
        dataset,
        FLAGS.split,
        os.path.join(FLAGS.keypoint_matching_res_dir, FLAGS.split),
        output_dir,
        cnfg,
        use_detections=(FLAGS.detector != ""),
    )

    exit(0)
