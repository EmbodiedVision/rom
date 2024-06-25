import os
import argparse
from datetime import datetime

"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

# Disable TF info messages (needs to be done before tf import)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from utils.helper_funcs import *
from utils.tf_funcs import *


# GPU should not be allocated entirely at beginning
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if len(gpu_devices) > 0:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)


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
        "--model", default="romnet-vizfeat", help="Model name [e.g. romnet-vizfeat]"
    )
    parser.add_argument(
        "--viz_feat_name", default="", help="Output directory name for viz features"
    )
    return parser.parse_args()


# --------------------------------------------------
# --- Main Evaluation


def main(net, cnfg, data):
    time_start_str = "{:%d.%m_%H:%M}".format(datetime.now())
    print("**** start: %s ****" % (time_start_str))

    # Create dataset iterator for testing + ops directory
    _, iterator = get_data_iterator(data)
    ops = {
        "iterator": iterator,
    }

    num_batches = int(data.get_size())
    b_mod = 100 if (num_batches < 1500) else 1000

    def init_list(ref_list):
        return [
            [
                [[] for frame_id in range(len(ref_list[scene_id][cam_id]))]
                for cam_id in range(len(ref_list[scene_id]))
            ]
            for scene_id in range(len(ref_list))
        ]

    obj_feats_max = init_list(data.obj_info["id"])

    seen = []
    last_scene_id = -1
    for batch_id in range(num_batches):

        if batch_id % b_mod == 0:
            print("Current image/total image num: %d/%d" % (batch_id, num_batches))

        input_batch = next(ops["iterator"])
        input_batch = net.get_input(input_batch)

        scene_id = input_batch["scan_ids"][
            0
        ].numpy()  # index 0 as first image is considered (same for scan_id anyway)
        cam_id = input_batch["cam_ids"][0].numpy()
        frame_id = input_batch["frame_ids"][0].numpy()

        cur = (scene_id, cam_id, frame_id)
        if cur[0] != last_scene_id:

            print(f"--> New scene: {scene_id}")
            last_scene_id = scene_id

        if cur not in seen:
            seen.append(cur)
            print(f"Current scene: {scene_id}, cam: {cam_id}, frame: {frame_id}")
        else:
            print(
                f"[WARNING] Current scene has been seen already! - scene: {scene_id}, cam: {cam_id}, frame: {frame_id}"
            )

        # compute feats
        output = net(input_batch)
        viz_feat_obj_max = output["viz_feat_obj_max"]

        obj_feats_max[scene_id][cam_id][frame_id] = viz_feat_obj_max.numpy()

    np.save(FEAT_DATA_PATH + "_objects.npy", obj_feats_max)
    print(f"[done] - total images: {len(seen)}")


# --------------------------------------------------


if __name__ == "__main__":

    FLAGS = parse_arguments()

    # Load basic model and config file
    print("Load model and config file..")
    model_base_name, model_ext_name = FLAGS.model.split("-")
    cnfg_name = "cnfg_{}_{}_pretrained_feats".format(model_base_name, FLAGS.dataset)
    model_file = os.path.join("models", model_base_name + ".py")
    model_module = importlib.import_module("." + model_base_name, "models")
    data_provider_file = os.path.join("utils", "data_provider.py")
    data_provider_module = importlib.import_module(".data_provider", "utils")
    cnfg, cnfg_file = load_cnfg(cnfg_name, base_dir="config")

    model = model_module.get_model(cnfg, model_ext_name)

    for split in ["val", "test", "train"]:
        print("->>", split)
        FEAT_DATA_DIR = os.path.join(FLAGS.data_dir, split, "precomputed_viz_feats")
        if not os.path.exists(FEAT_DATA_DIR):
            os.mkdir(FEAT_DATA_DIR)
        FEAT_DATA_PATH = os.path.join(FEAT_DATA_DIR, FLAGS.viz_feat_name)
        if os.path.exists(FEAT_DATA_PATH + "_objects.npy"):
            print(
                "File already exists, delete this first before re-computing features ({})".format(
                    FEAT_DATA_PATH
                )
            )
            continue

        # Load data
        print("Load data..")
        dataset = data_provider_module.get_dataset(
            FLAGS.data_dir, "direct", split, cnfg["data"]
        )

        main(model, cnfg, dataset)

    exit(0)
