"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

import os
import argparse
import numpy as np
import math
import random as rnd
import csv
import h5py
import pickle

from rel_obj_match.utils.helper_funcs import check_if_dir_exists
from rel_obj_match.utils.data_info import *


MIN_N_INTERESTING_OBJ = 3
MIN_OBJ_SIZE = 25
CLS_LABELING = "NYU40"

IMG_SIZE_ORG = (768, 1024)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/is/rg/ev/scratch/celich/data/Hypersim/Hypersim",
        help="Path to HyperSim data dir: <...>/Hypersim",
    )
    parser.add_argument(
        "--meta_data_dir",
        default="/is/sg2/celich/projects/datasets/ml-hypersim/evermotion_dataset/analysis",
        help="Path to meta data: <...>/ml-hypersim/evermotion_dataset/analysis",
    )
    parser.add_argument(
        "--output_name_ext",
        default="_Processed",
        help="Name extension for output directory: <data_dir><output_name_ext>",
    )
    parser.add_argument(
        "--split", default="train", help="Split from train/val/test [default: train]"
    )
    parser.add_argument(
        "--subset",
        default=False,
        type=bool,
        help="Process subset only for debugging purpose [default: False]",
    )
    return parser.parse_args()


def get_scenes(data_dir, meta_data_dir, out_dir, split, subset):
    print("Scene Names")

    val_scene_names = []
    val_scene_cam_names = {}
    n_scenes_total = 0
    n_scenes_valid = 0
    with open(
        os.path.join(meta_data_dir, "metadata_camera_trajectories.csv"), newline=""
    ) as csvfile:
        meta_data = csv.reader(csvfile, delimiter=",")
        header = next(meta_data)
        for row in meta_data:
            n_scenes_total += 1
            scene_cam_name, _, _, _, _, _, _, scene_type, notes, _ = row
            notes = notes.lower()
            if (
                "copyrighted" in notes
                or "weird" in notes
                or "unrealistic" in notes
                or "empty" in notes
            ):
                # print('  Ignore trajectory:', scene_cam_name, scene_type, notes)
                continue
            if "BAD" in scene_type:
                # print('   - excluded (2):', scene_cam_name, scene_type, '--', row)
                continue

            n_scenes_valid += 1
            scene_name = scene_cam_name.split("_cam")[0]
            cam_name = "cam" + scene_cam_name.split("_cam")[1]
            if scene_name not in val_scene_names:
                val_scene_names.append(scene_name)
                val_scene_cam_names[scene_name] = []
            val_scene_cam_names[scene_name].append(cam_name)
    print(
        f"  - TOTAL: {n_scenes_total} scenes, "
        f"{n_scenes_valid} are valid, "
        f"{len(val_scene_names)} distinct scenes"
    )

    final_scene_names = []
    final_scene_cams = {}
    final_scene_frames = {}
    final_scene_mpa = {}
    n_scenes_final = 0
    n_frames_final = 0
    n_frames_per_scene_final = []
    with open(
        os.path.join(meta_data_dir, "metadata_images_split_scene_v1.csv"), newline=""
    ) as csvfile:
        meta_data = csv.reader(csvfile, delimiter=",")
        header = next(meta_data)
        for row in meta_data:
            (
                scene_name,
                cam_name,
                frame_id,
                included_in_public_release,
                _,
                split_name,
            ) = row
            if (
                not included_in_public_release == "True"
                or scene_name not in val_scene_names
                or split_name != split
            ):
                continue

            if scene_name not in final_scene_names:
                n_scenes_final += 1
                final_scene_names.append(scene_name)
                final_scene_cams[scene_name] = []
                final_scene_frames[scene_name] = {}
                n_frames_per_scene_final.append(0)
            if cam_name not in final_scene_cams[scene_name]:
                final_scene_cams[scene_name].append(cam_name)
                final_scene_frames[scene_name][cam_name] = []

            n_frames_final += 1
            n_frames_per_scene_final[-1] += 1
            final_scene_frames[scene_name][cam_name].append(frame_id)

    print(f"  - final for split {split}:")
    print(f"      #scenes: {n_scenes_final}, #frames (total): {n_frames_final}")

    if subset:  # only subset of scenes for first experiments:
        subset_scene_name = final_scene_names[0]
        final_scene_names = [subset_scene_name]
        final_scene_cams = {subset_scene_name: final_scene_cams[subset_scene_name]}
        final_scene_frames = {subset_scene_name: final_scene_frames[subset_scene_name]}

    # get info for scene specific meters_per_asset_unit
    for scene_name in final_scene_names:
        with open(
            os.path.join(data_dir, scene_name, "_detail", "metadata_scene.csv"),
            newline="",
        ) as metadata_scene_csvfile:
            meta_scene = csv.reader(metadata_scene_csvfile, delimiter=",")
            for row in meta_scene:
                if row[0] == "meters_per_asset_unit":
                    meters_p_asset_u = float(row[1])
                    final_scene_mpa[scene_name] = meters_p_asset_u

    # save scene names & data
    with open(os.path.join(out_dir, "scene_names.txt"), "w") as f:
        for item in final_scene_names:
            f.write("%s\n" % item)

    scene_info_dict = {
        "cams": final_scene_cams,
        "frames_all": final_scene_frames,
        "meters_per_asset_unit": final_scene_mpa,
    }
    np.save(os.path.join(out_dir, "scene_info.npy"), scene_info_dict)

    return final_scene_names, scene_info_dict


def get_cam_info(data_dir, out_dir):

    print("Get camera info...")
    with open(os.path.join(out_dir, "scene_names.txt"), "r") as f:
        scene_names = [name.replace("\n", "") for name in f]
    scene_info = np.load(
        os.path.join(out_dir, "scene_info.npy"), allow_pickle=True
    ).item()
    scene_cams = scene_info["cams"]
    scene_frames = scene_info["frames_all"]

    # Get Intrinsic
    # cf. https://github.com/apple/ml-hypersim/blob/master/code/python/tools/scene_generate_images_bounding_box.py
    height, width = [768, 1024]  # constant over entire dataset

    # fov_x and fov_y need to match the _vray_user_params.py that was used to generate the images
    fov_x = np.pi / 3.0  # ->60' [degrees]
    fov_y = 2.0 * np.arctan(height * np.tan(fov_x / 2.0) / width)

    f_x = height / (2.0 * np.tan(fov_y / 2.0))
    f_y = width / (2.0 * np.tan(fov_x / 2.0))
    assert np.abs(f_y - f_x) < 1e-8  # otherwise, make sure that this still works
    c_x, c_y = [width / 2.0, height / 2.0]
    cam_intr = np.asarray([[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]])

    # Get extrinsic
    cams_extr = []
    cams_intr = []

    for scene_name in scene_names:

        cams_intr.append(cam_intr)

        cams_extr.append([])
        cams = scene_cams[scene_name]
        for cam in cams:
            frames = scene_frames[scene_name][cam]

            cams_extr[-1].append([])
            cam_path = os.path.join(
                data_dir, scene_name, "_detail", cam, "camera_keyframe"
            )
            cam_orientations = h5py.File((cam_path + "_orientations.hdf5"), "r")[
                "dataset"
            ]  # (N_img, 3, 3)
            cam_positions = h5py.File((cam_path + "_positions.hdf5"), "r")[
                "dataset"
            ]  # (N_img, 3)

            for frame_id in frames:
                cur_cam_ori = cam_orientations[int(frame_id)]
                cur_cam_pos = cam_positions[int(frame_id)]
                extrinsic = np.concatenate(
                    [
                        np.concatenate(
                            [
                                np.transpose(cur_cam_ori),
                                -np.transpose(cur_cam_ori)
                                @ np.reshape(cur_cam_pos, (3, 1)),
                            ],
                            axis=1,
                        ),
                        np.asarray([[0.0, 0.0, 0.0, 1.0]]),
                    ],
                    axis=0,
                )  # world to cam transformation
                cams_extr[-1][-1].append(extrinsic)  # latest scene, latest cam
            cams_extr[-1][-1] = np.asarray(cams_extr[-1][-1])

    cam_info_dict = {"intr": cams_intr, "extr": cams_extr}
    np.save(os.path.join(out_dir, "frame_cam_info.npy"), cam_info_dict)

    return cam_info_dict


def get_frame_obj_info(data_dir, out_dir):
    print("Get frame-wise object info...")
    with open(os.path.join(out_dir, "scene_names.txt"), "r") as f:
        scene_names = [name.replace("\n", "") for name in f]
    scene_info = np.load(
        os.path.join(out_dir, "scene_info.npy"), allow_pickle=True
    ).item()
    scene_cams = scene_info["cams"]
    scene_frames = scene_info["frames_all"]
    cam_info = np.load(
        os.path.join(out_dir, "frame_cam_info.npy"), allow_pickle=True
    ).item()

    obj_info = {
        "id": [],  # scene-specific object id (from instance map)
        "cls": [],
        "cntr3d": [],  # in cam coord system
        "bb2d": [],
        "cntr2d": [],  # projected center
        "offset": [],  # offset between 2d bb center and projected 3d object center
        "dist": [],  # distance from object center to camera
    }
    scene_frames_updated = {}

    extrinsic_cams = cam_info["extr"]
    intrinsic_cams = cam_info["intr"]

    def get_bb_from_inst(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        x_min, x_max = np.where(rows)[0][[0, -1]]
        y_min, y_max = np.where(cols)[0][[0, -1]]
        return x_min, y_min, x_max, y_max

    def is_meaningful_obj(lbl, bb2d):
        return (
            lbl not in get_subset_categories(CLS_LABELING, subset="structure")
        ) and (bb2d[0] <= bb2d[2] - MIN_OBJ_SIZE and bb2d[1] <= bb2d[3] - MIN_OBJ_SIZE)

    def _coord_world2cam(pnts_world, extr, intr):
        n_pnts = pnts_world.shape[0]
        pnts_world = np.concatenate([pnts_world, np.ones((n_pnts, 1))], axis=-1)
        pnts_world = np.expand_dims(pnts_world, axis=-1)  # => (N_pnts, 4, 1)

        pnts_cam = extr @ pnts_world
        pnts_cam = pnts_cam[:, :3]

        pnts_proj = intr @ ([[1.0], [1.0], [-1.0]] * pnts_cam)  # for own intrinisc
        pnts_proj = pnts_proj[:, :2] / pnts_proj[:, 2:3]
        pnts_proj = np.squeeze(pnts_proj, axis=-1).astype(int)
        pnts_proj = np.stack(
            [img_height - pnts_proj[:, 1], pnts_proj[:, 0]], axis=-1
        )  # transfer to np-coord system

        pnts_cam = np.squeeze(pnts_cam, axis=-1)

        return pnts_cam, pnts_proj

    for scene_id, scene_name in enumerate(scene_names):
        scene_frames_updated[scene_name] = {}

        for k in obj_info.keys():
            obj_info[k].append([])

        cams = scene_cams[scene_name]
        for cam_id, cam in enumerate(cams):
            frames = [int(f) for f in scene_frames[scene_name][cam]]
            scene_frames_updated[scene_name][cam] = []

            for k in obj_info.keys():
                obj_info[k][-1].append([])

            mesh_path = os.path.join(
                data_dir,
                scene_name,
                "_detail",
                "mesh",
                "metadata_semantic_instance_bounding_box_object_aligned_2d",
            )
            geometry_path = os.path.join(
                data_dir,
                scene_name,
                "images",
                "scene_" + cam + "_geometry_hdf5",
                "frame",
            )
            bb3d_positions = h5py.File((mesh_path + "_positions.hdf5"), "r")[
                "dataset"
            ]  # (N_obj, 3)
            bb3d_extents = h5py.File((mesh_path + "_extents.hdf5"), "r")[
                "dataset"
            ]  # (N_obj, 3)  # -> for testing 3d bb projection
            bb3d_orientations = h5py.File((mesh_path + "_orientations.hdf5"), "r")[
                "dataset"
            ]  # (N_obj, 3, 3)  # -> for testing 3d bb projection

            scene_extrs = extrinsic_cams[scene_id][cam_id]
            scene_intr = intrinsic_cams[scene_id]

            img_width, img_height = [1024, 768]  # constant over entire dataset

            n_invalid_obj = 0
            n_frames_with_invalid_obj = 0

            for frame_id, frame_name in enumerate(frames):

                semantic = np.asarray(
                    h5py.File(
                        (geometry_path + ".{:04d}.semantic.hdf5".format(frame_name)),
                        "r",
                    )["dataset"]
                )  # (H, W), <int.16>
                instance = np.asarray(
                    h5py.File(
                        (
                            geometry_path
                            + ".{:04d}.semantic_instance.hdf5".format(frame_name)
                        ),
                        "r",
                    )["dataset"]
                )  # (H, W), <int.16>

                frame_extr = scene_extrs[frame_id]

                frame_obj_info = {k: [] for k in obj_info.keys()}
                frame_object_ids = np.unique(instance)

                for (
                    obj_id
                ) in (
                    frame_object_ids
                ):  # only iterate over instances in current image/view

                    if obj_id == -1:
                        continue

                    # get obj semantic
                    sem = semantic[instance == obj_id]
                    sem = sem[sem > 0]
                    if len(np.unique(sem)) != 1:
                        if len(np.unique(sem)) > 1:
                            print(
                                f"\t\t invalid semseg for frame_id: {frame_id}, obj_id: {obj_id}, "
                                f"unique sem: {np.unique(sem)}, unique sem, v2: {np.unique(semantic[(instance == obj_id)])}, "
                                f"sum(inst=obj_id): {np.sum(instance == obj_id)}"
                            )
                        continue
                    sem = sem[0] - 1

                    # get 2D bb
                    msked_img = instance == obj_id
                    bb2d = get_bb_from_inst(msked_img)
                    size2d = [bb2d[2] - bb2d[0], bb2d[3] - bb2d[1]]
                    bb2d_cntr = np.asarray(
                        [(bb2d[0] + bb2d[2]) / 2.0, (bb2d[1] + bb2d[3]) / 2.0]
                    )

                    if not is_meaningful_obj(sem, bb2d):
                        continue

                    cntr3d = bb3d_positions[obj_id]  # world coordinates
                    if np.isinf(cntr3d).any():
                        print(
                            f"\t\t invalid bb_positions (inf) for frame_id: {frame_id}, obj_id: {obj_id}, pos {pos}"
                        )
                        continue

                    cntr3d_cam, cntr2d_proj = _coord_world2cam(
                        np.expand_dims(cntr3d, axis=0), frame_extr, scene_intr
                    )
                    cntr3d_cam = cntr3d_cam[0]
                    cntr2d_proj = cntr2d_proj[0]

                    dist = np.sqrt(
                        cntr3d_cam[0] ** 2 + cntr3d_cam[1] ** 2 + cntr3d_cam[2] ** 2
                    )
                    offset = cntr2d_proj - bb2d_cntr
                    offset_norm = offset / size2d

                    frame_obj_info["id"].append(obj_id)
                    frame_obj_info["cls"].append(sem)
                    frame_obj_info["cntr3d"].append(cntr3d_cam)  # camera coord systems
                    frame_obj_info["bb2d"].append(bb2d)
                    frame_obj_info["cntr2d"].append(cntr2d_proj)
                    frame_obj_info["offset"].append(offset_norm)
                    frame_obj_info["dist"].append(dist)

                # filter out frames with only few/ none interesting objects(only few/no objects visible)
                if len(frame_obj_info["id"]) < MIN_N_INTERESTING_OBJ:
                    continue
                scene_frames_updated[scene_name][cam].append(frame_name)

                for k in obj_info.keys():
                    obj_info[k][-1][-1].append(frame_obj_info[k])

            n_frames_org = len(scene_frames[scene_name][cam])
            n_frames_updated = len(scene_frames_updated[scene_name][cam])
            print(
                f"  -> {scene_name} {cam} #frames: {n_frames_updated}/ {n_frames_org}"
            )
            if n_frames_with_invalid_obj > 0:
                print(
                    f"      => invalid projection, #frames: {n_frames_with_invalid_obj}, "
                    f"#objects (total): {n_invalid_obj}"
                )
    np.save(os.path.join(out_dir, "frame_obj_anno.npy"), obj_info)

    scene_info["frames"] = scene_frames_updated
    np.save(os.path.join(out_dir, "scene_info.npy"), scene_info)

    return obj_info, scene_frames_updated


def get_frame_rgb_names(data_dir, out_dir):
    print("Get frame-wise rgb names...")
    with open(os.path.join(out_dir, "scene_names.txt"), "r") as f:
        scene_names = [name.replace("\n", "") for name in f]
    scene_info = np.load(
        os.path.join(out_dir, "scene_info.npy"), allow_pickle=True
    ).item()
    scene_cams = scene_info["cams"]
    scene_frames = scene_info["frames"]

    rgb_names = []

    for scene_name in scene_names:
        cams = scene_cams[scene_name]
        rgb_names.append([])
        for cam in cams:
            frames = scene_frames[scene_name][cam]

            rgb_dir = os.path.join(
                data_dir, scene_name, "images", "scene_" + cam + "_final_preview"
            )
            rgb_imgs_all = os.listdir(rgb_dir)
            rgb_imgs_all = [rgb for rgb in rgb_imgs_all if "tonemap.jpg" in rgb]
            rgb_imgs_all = [
                rgb_name.replace("tonemap.jpg", "") for rgb_name in rgb_imgs_all
            ]

            rgb_imgs_sub = [
                rgb_name
                for rgb_name in rgb_imgs_all
                if int(rgb_name.split(".")[1]) in frames
            ]

            rgb_names[-1].append(rgb_imgs_sub)

    with open(os.path.join(out_dir, "frame_rgb_names.pickle"), "wb") as fp:
        pickle.dump(rgb_names, fp)

    return rgb_names


def get_global_obj_info(out_dir):
    print("Get global object info...")
    with open(os.path.join(out_dir, "scene_names.txt"), "r") as f:
        scene_names = [name.replace("\n", "") for name in f]
    scene_info = np.load(
        os.path.join(out_dir, "scene_info.npy"), allow_pickle=True
    ).item()
    scene_cams = scene_info["cams"]
    frame_obj_info = np.load(
        os.path.join(out_dir, "frame_obj_anno.npy"), allow_pickle=True
    ).item()
    frame_obj_ids = frame_obj_info["id"]
    frame_obj_cls = frame_obj_info["cls"]

    global_obj_info = {
        "n_cls": np.asarray([0 for _ in range(len(get_class_names(CLS_LABELING)))]),
        "id2sem": [],
    }

    total_n_scenes = 0
    total_n_frames = 0
    total_n_obj = 0

    for scene_id, scene_name in enumerate(scene_names):
        cur_n_frames = 0
        global_obj_info["id2sem"].append({})
        cams = scene_cams[scene_name]
        for cam_id, cam in enumerate(cams):
            cur_n_frames += len(frame_obj_ids[scene_id][cam_id])
            for frame_id in range(len(frame_obj_ids[scene_id][cam_id])):
                n_obj_per_class = [0 for _ in range(len(get_class_names(CLS_LABELING)))]
                for idx in range(len(frame_obj_ids[scene_id][cam_id][frame_id])):
                    obj_id = frame_obj_ids[scene_id][cam_id][frame_id][idx]
                    obj_cls = frame_obj_cls[scene_id][cam_id][frame_id][idx]
                    n_obj_per_class[obj_cls] += 1
                    if not (
                        obj_id in global_obj_info["id2sem"][-1].keys()
                    ):  # same for all cams, thus no differentiation required here
                        global_obj_info["id2sem"][-1][obj_id] = obj_cls
        if cur_n_frames > 0:
            total_n_scenes += 1
            total_n_frames += cur_n_frames
            total_n_obj += len(global_obj_info["id2sem"][-1].keys())

    print(
        "TOTAL - #scenes:",
        total_n_scenes,
        "#frames:",
        total_n_frames,
        "#objects:",
        total_n_obj,
    )

    for scene_id, scene_name in enumerate(scene_names):

        id2sem = global_obj_info["id2sem"][scene_id]

        for obj_id, cls in id2sem.items():
            global_obj_info["n_cls"][cls] += 1

    np.save(os.path.join(out_dir, "global_obj_info.npy"), global_obj_info)
    return global_obj_info


def update_dict_cam(file_name, out_dir):
    print("Update camera info..")
    with open(os.path.join(out_dir, "scene_names.txt"), "r") as f:
        scene_names = [name.replace("\n", "") for name in f]

    scene_info = np.load(
        os.path.join(out_dir, "scene_info.npy"), allow_pickle=True
    ).item()
    scene_cams = scene_info["cams"]
    scene_frames_old = scene_info["frames_all"]
    scene_frames_new = scene_info["frames"]

    data_info = np.load(os.path.join(out_dir, file_name + ".npy"), allow_pickle=True)[
        ()
    ]
    if "cam_all" in data_info.keys():
        data_info = data_info["cam_all"]
    else:
        data_info["cam_all"] = data_info.copy()

    data_updated = {"cam_all": data_info["cam_all"], "intr": [], "extr": []}

    for scene_idx, scene_name in enumerate(scene_names):
        data_updated["extr"].append([])
        data_updated["intr"].append(data_info["intr"][scene_idx])

        cams = scene_cams[scene_name]
        for cam_idx, cam in enumerate(cams):
            data_updated["extr"][-1].append([])
            data_scene_extr = data_info["extr"][scene_idx][cam_idx]
            for frame_idx, frame_name in enumerate(scene_frames_new[scene_name][cam]):
                frame_idx_old = scene_frames_old[scene_name][cam].index(str(frame_name))
                data_updated["extr"][-1][-1].append(data_scene_extr[frame_idx_old])
    np.save(os.path.join(out_dir, file_name + ".npy"), data_updated)


def get_frame_pairs(data_dir, out_dir):

    print("Get frame pairs...")
    scene_info = np.load(
        os.path.join(out_dir, "scene_info.npy"), allow_pickle=True
    ).item()
    scene_cams = scene_info["cams"]  # only names
    scene_frames = scene_info["frames"]

    frame_obj_info = np.load(
        os.path.join(out_dir, "frame_obj_anno.npy"), allow_pickle=True
    ).item()
    cam_info = np.load(
        os.path.join(out_dir, "frame_cam_info.npy"), allow_pickle=True
    ).item()
    cam_info = cam_info["extr"]

    scene_names = [scene_name for scene_name in scene_frames.keys()]

    def valid_extrinsics(extr, extr_r, scene_name, cam, frame):
        if (
            np.sum(extr_r @ extr[:3, :3]) > 3.1 or np.sum(extr_r @ extr[:3, :3]) < 2.9
        ):  # there seem to be wrong rotation annotations in the dataset
            tmp_name = f"{scene_name}_{cam}_{frame}"
            if tmp_name not in tmp_skipped_scene_frames:
                tmp_skipped_scene_frames.append(tmp_name)
                print(
                    f"  - (1) skip scene: {scene_name}, cam: {cam}, frame: {frame} "
                    f"due to bad extrinsic (rot) matrix - \n",
                    extr,
                )
            return False

        return True

    # wrt objects
    obj_dist_diff = []
    obj_angle = []
    obj_dist_all = []
    obj_angle_all = []

    matching_frames = (
        []
    )  # [(scene_id, cam_id1, frame_id1, cam_id2, frame_id2, level_cam, level_obj)]
    matching_frames_sample_obj = []

    n_levels = 3
    n_level_obj = [0 for _ in range(n_levels)]

    tmp_skipped_scene_frames = []
    tmp_n_img_with_corresp = 0

    for scene_id, scene_name in enumerate(scene_names):
        print(f"{scene_id+1}/{len(scene_names)}, {scene_name}")

        cams = scene_cams[scene_name]
        mpa = scene_info["meters_per_asset_unit"][scene_name]  # same for entire scene

        mesh_path = os.path.join(
            data_dir,
            scene_name,
            "_detail",
            "mesh",
            "metadata_semantic_instance_bounding_box_object_aligned_2d",
        )
        bb_positions = h5py.File((mesh_path + "_positions.hdf5"), "r")[
            "dataset"
        ]  # (N_obj, 3)

        for cam_id1, cam1 in enumerate(cams):

            for frame_id1, frame1 in enumerate(scene_frames[scene_name][cam1]):

                cur_matches = []

                extr1 = cam_info[scene_id][cam_id1][frame_id1]
                extr1_r = np.transpose(extr1[:3, :3])
                extr1_t = (
                    extr1_r @ -extr1[:3, 3:]
                )  # that's the cam pos in world coord system

                if not valid_extrinsics(
                    extr1[:3, :3], extr1_r, scene_name, cam1, frame1
                ):
                    continue

                obj1_ids = frame_obj_info["id"][scene_id][cam_id1][frame_id1]

                tmp_match = False  # check whether some matching frame was found

                for cam_id2, cam2 in [
                    [cam_id1, cam1]
                ]:  # within trajectory only (test)  #enumerate(cams):
                    for frame_id2, frame2 in enumerate(scene_frames[scene_name][cam2]):
                        if cam_id2 == cam_id1 and frame_id2 == frame_id1:
                            continue

                        valid = True  # use this flag to ignore invalid frame-pairs for novel data pairings, but still run procedure until end

                        obj2_ids = frame_obj_info["id"][scene_id][cam_id2][frame_id2]

                        # -- Overlapping Objects
                        obj_ids_common = list(set(obj1_ids) & set(obj2_ids))
                        if len(obj_ids_common) == 0:
                            # print(f'  - (4) skip image pair: {scene_name}, cam: {cam1}/{cam2}, frame_id: {frame1}/{frame2} '
                            #       f'due to empty overlapping object set')
                            valid = False

                        # -- Cam Difference; keep this for removing scenes with bad camera settings
                        extr2 = cam_info[scene_id][cam_id2][frame_id2]

                        extr2_r_T = extr2[:3, :3]
                        extr2_r = np.transpose(extr2[:3, :3])
                        extr2_t = extr2_r @ -extr2[:3, 3:]

                        if not valid_extrinsics(
                            extr2[:3, :3], extr2_r, scene_name, cam2, frame2
                        ):
                            valid = False

                        cam_delta_t = np.sqrt(np.sum((extr1_t - extr2_t) ** 2)) * mpa
                        # http://www.boris-belousov.net/2016/12/01/quat-dist/
                        diff_R = extr1_r @ extr2_r_T
                        cam_delta_r = np.arccos((np.trace(diff_R) - 1) / 2)
                        cam_delta_r = cam_delta_r / (2 * math.pi) * 360

                        if (np.trace(diff_R) - 1) / 2 < -1.0:
                            print(
                                f"  - (3) skip scene: {scene_name}, cam: {cam1}/{cam2}, frame: {frame1}/{frame2} "
                                f"due to numerical issues => trace(diff_R)=",
                                np.trace(diff_R),
                            )
                            valid = False
                        if np.isnan(cam_delta_t):
                            print(
                                f"  - (4) skip scene: {scene_name}, cam: {cam1}/{cam2}, frame: {frame1}/{frame2} "
                                f"due to delta_t(cam)=NAN",
                                cam_delta_t,
                                cam_delta_r,
                            )  # hasn't occured yet
                            valid = False
                        if np.isnan(cam_delta_r):
                            print(
                                f"  - (4) skip scene: {scene_name}, cam: {cam1}/{cam2}, frame: {frame1}/{frame2} "
                                f"due to delta_r(cam)=NAN",
                                cam_delta_t,
                                cam_delta_r,
                            )  # (likely -> checked for test set) due to numerical instabilities
                            valid = False

                        # -- Object Difference
                        obj_pos = [
                            bb_positions[i] for i in obj_ids_common
                        ]  # obj ids are global ids

                        obj_cam1_vec = [
                            extr1_t[:, 0] - obj_pos_i for obj_pos_i in obj_pos
                        ]
                        obj_cam2_vec = [
                            extr2_t[:, 0] - obj_pos_i for obj_pos_i in obj_pos
                        ]

                        dist1 = [np.sqrt(np.sum(np.square(v))) for v in obj_cam1_vec]
                        dist2 = [np.sqrt(np.sum(np.square(v))) for v in obj_cam2_vec]
                        obj_cam_dist_diff = [
                            np.abs(dist1[i] - dist2[i]) * mpa for i in range(len(dist1))
                        ]
                        obj_cam_angle = [
                            np.arccos(
                                np.sum(obj_cam1_vec[i] * obj_cam2_vec[i])
                                / (dist1[i] * dist2[i])
                            )
                            / (2 * math.pi)
                            * 360
                            for i in range(len(dist1))
                        ]

                        avg_obj_cam_dist_diff = np.mean(obj_cam_dist_diff)
                        avg_obj_cam_angle = np.mean(obj_cam_angle)

                        if valid:
                            obj_dist_diff.append(avg_obj_cam_dist_diff)
                            obj_angle.append(avg_obj_cam_angle)

                            obj_dist_all.append(obj_cam_dist_diff)
                            obj_angle_all.append(obj_cam_angle)

                            obj_t_th = [4.0, 8.0]
                            obj_r_th = [45.0, 90.0]
                            level_obj = len(obj_t_th)
                            for i in range(len(obj_t_th) - 1, -1, -1):
                                if (
                                    avg_obj_cam_dist_diff <= obj_t_th[i]
                                    and avg_obj_cam_angle <= obj_r_th[i]
                                ):
                                    level_obj = i
                            n_level_obj[level_obj] += 1

                            match = [
                                scene_id,
                                cam_id1,
                                frame_id1,
                                cam_id2,
                                frame_id2,
                                level_obj,
                                avg_obj_cam_dist_diff,
                                avg_obj_cam_angle,
                            ]
                            matching_frames.append(match)
                            cur_matches.append(match)

                        if not tmp_match:
                            tmp_match = True
                            tmp_n_img_with_corresp += 1

                for lvl in range(n_levels):
                    cur_matches = np.asarray(cur_matches)
                    if len(cur_matches) == 0:
                        continue

                    cur_matches_l = cur_matches[cur_matches[:, -3] == lvl]
                    if len(cur_matches_l) > 1:
                        smpl_match_id = rnd.choice(range(len(cur_matches_l)))
                        matching_frames_sample_obj.append(cur_matches_l[smpl_match_id])

    print(
        "Total #matches:",
        len(matching_frames),
        ", # unique frames: ",
        tmp_n_img_with_corresp,
    )
    matching_frames = np.asarray(matching_frames)
    print("Difficulty wrt. object: ", n_level_obj, "total:", np.sum(n_level_obj))

    print("(Subsample) Total #matches:", "(obj)", len(matching_frames_sample_obj))
    matching_frames_sample_obj = np.asarray(matching_frames_sample_obj)
    tmp_n_obj = [
        (matching_frames_sample_obj[matching_frames_sample_obj[:, -3] == lvl]).shape[0]
        for lvl in range(n_levels)
    ]
    print("Difficulty wrt. object: ", tmp_n_obj)

    np.save(os.path.join(out_dir, "matching_frames.npy"), matching_frames)
    np.save(
        os.path.join(out_dir, "matching_frames_sample_obj.npy"),
        matching_frames_sample_obj,
    )


if __name__ == "__main__":

    args = parse_arguments()

    check_if_dir_exists(args.data_dir)

    output_dir = args.data_dir + args.output_name_ext
    if args.subset:
        output_dir += "_subset"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created output directory {}".format(output_dir))
    else:
        print(
            "Output directory exists; potentially overwriting contents ({}).".format(
                output_dir
            )
        )

    if args.split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [args.split]

    for split in splits:
        output_split_dir = os.path.join(output_dir, split)
        if not os.path.exists(output_split_dir):
            os.makedirs(output_split_dir)
        else:
            print(f"  skip split {split} as it already exists")
            continue

        if args.subset:
            split = "train"

        print("Load data from: ", args.data_dir)
        # scene_names   = [<str> for each scene],
        # scene_info    = {'cams':{},
        #                  'frames_all':{scene_name:{cam_name:[frame_name_short]}},
        #                  'meters_per_asset_unit':{scene_name:<int>}
        scene_names, scene_info = get_scenes(
            args.data_dir, args.meta_data_dir, output_split_dir, split, args.subset
        )

        # cam_info      = {'intr': (3, 3),
        #                  'extr': [(3, 3) for each scene]: [<float> for each scene]}
        cam_info = get_cam_info(args.data_dir, output_split_dir)
        # obj_anno      = {'id'/'cls'/'cntr3d'/'bb2d'/'cntr2d'/'offset'/'dist':
        #                       [[[(?) for each obj] for each frame] for each scene]}
        obj_anno, _ = get_frame_obj_info(args.data_dir, output_split_dir)

        # rgb_names     = [[[<str> for each obj] for each frame] for each scene]
        rgb_names = get_frame_rgb_names(args.data_dir, output_split_dir)

        # obj_info_global = {'n_cls': [<int> for each class],
        #                    'id2sem': [{<int>: <int> } for each scene]}
        obj_info_global = get_global_obj_info(
            output_split_dir
        )  # {'N_cls', 'id2sem'}  , previous: 'avg_obj_size'

        # cam_info      = {..., 'cam_all':{'intr': (3, 3),
        #                              'extr': [(3, 3) for each scene]: [<float> for each scene]}}
        update_dict_cam("frame_cam_info", output_split_dir)

        # (N_matches, 8[scene_id, cam_id1, frame_id1, cam_id2, frame_id2, level_cam, cam_delta_t, cam_delta_r])
        get_frame_pairs(args.data_dir, output_split_dir)

    exit(0)
