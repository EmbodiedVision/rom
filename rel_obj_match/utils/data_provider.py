"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

import os
import numpy as np
import itertools
import random
import cv2
from PIL import Image
import pickle
import tensorflow as tf
from itertools import product

from utils.data_info import *
from utils.data_augmentation import DataAugmetation

# -> for testing only
import utils.viz as viz
from utils import tools_eval
from utils.helper_funcs import load_cnfg


# --------------------------------------------------

DATASET_SPLITS = ["train", "val", "test"]


# --------------------------------------------------


class DataSet:

    def __init__(self, data_dir, split, cnfg):
        """
        Initialize dataset object
        :param data_dir:    string, directory of data
        :param split:       string, split of dataset (from DATASET_SPLITS)
        :param cnfg:        dict, dataset configuration
        """

        if not os.path.exists(data_dir):
            print("DataSet.__init__(): Directory {} does not exist.".format(data_dir))
            for _ in range(4):
                data_dir = os.path.dirname(data_dir)
                print(data_dir, os.path.exists(data_dir))
                if os.path.exists(data_dir):
                    print(os.listdir(data_dir))
                    break
                if data_dir == "":
                    break
            exit(1)
        assert split in DATASET_SPLITS

        self.data_dir = os.path.join(data_dir, split)
        self.split = split

        self.name = cnfg["name"]

    # --------------------------------
    # --- General functions:

    def get_size(self):
        """
        :return:    int, number of data samples
        """

        print(len(self.ids))
        return len(self.ids)

    # --------------------------------
    # --- Prepare input for model:

    def prepare_input(self, idx, rgb_noise=False):
        """
        Convert data to input format for network
        :param idx:           int,        index of object sample
        :param rgb_noise:     bool,       indicates whether to apply noise on rgb input
        :return: smpl_id:         (), int16
                 samples:         (N, 4), float32   -> (x,y,z,d)
        """

        # scene_name = self.scene_names[idx]
        scene_id = np.int32(idx) * np.ones(1, dtype=np.int32)

        assert scene_id.dtype == np.int32

        return scene_id

    def generator(self):
        """
        :return:    Generator for data sample ids
        """
        for i in itertools.cycle(self.ids):
            yield (i)


# --------------------------------------------------


class DataSetHyperSim(DataSet):

    # --------------------------------
    # --- Initialization:
    def __init__(self, data_dir, split, cnfg):
        super(DataSetHyperSim, self).__init__(data_dir, split, cnfg)

        self.img_shape = cnfg["img_shape"]
        self.img_shape_obj = cnfg["img_shape_obj"]
        self.img_shape_org = cnfg["img_shape_org"]

        self.n_smpl_obj = cnfg["n_smpl_obj"]
        self.cls_labeling = cnfg["cls_labeling"]
        self.n_classes = cnfg["num_cls_sem_obj"]

        self.use_pretrained = cnfg["use_pretrained"]
        self.augmentation_types = cnfg["augmentation"]
        self.augmentation_module = DataAugmetation(cnfg["augmentation"])

        self.dist_factor = cnfg["dist_factor"]
        self.bs = 1

        self.scene_info = np.load(
            os.path.join(self.data_dir, "scene_info.npy"), allow_pickle=True
        )[()]
        data_dir_img = "_".join(data_dir.split("_")[:1])
        frames = self.scene_info["frames"]
        self.scene_names = [scene_name for scene_name in frames.keys()]
        self.cam_names = [
            [cam_name for cam_name in frames[s].keys()] for s in frames.keys()
        ]
        self.base_img_path = os.path.join(
            data_dir_img, "{}", "images", "scene_{}_final_preview", "{}tonemap.jpg"
        )
        with open(os.path.join(self.data_dir, "frame_rgb_names.pickle"), "rb") as fp:
            self.rgb_frame_names = pickle.load(fp)

        self.obj_info = np.load(
            os.path.join(self.data_dir, "frame_obj_anno.npy"), allow_pickle=True
        )[
            ()
        ]  # un-wrap saved dictionary

        # -> load image-pair indices
        self.ids_sub, _ = self._init_indexes(cnfg["lvl_difficulty"][split])
        if self.split == "train":
            self.n_ids_unique = self._count_unique_scene_ids()
        self.ids = np.arange(self.ids_sub.shape[0])

        camera_data = np.load(
            os.path.join(self.data_dir, "frame_cam_info.npy"), allow_pickle=True
        )[()]
        self.camera_intr = camera_data["intr"]
        self.camera_extr = camera_data["extr"]

        self.obj_viz_feats = None
        self.context_viz_feats = None
        self.obj_viz_feats_path = None
        if self.use_pretrained:
            self.obj_viz_feats = np.load(
                os.path.join(
                    self.data_dir, "precomputed_viz_feats", "basic_ResNet34_objects.npy"
                ),
                allow_pickle=True,
            )

        global_obj_info = np.load(
            os.path.join(self.data_dir, "global_obj_info.npy"), allow_pickle=True
        )[
            ()
        ]  # un-wrap saved dictionary
        self.lbl_count = global_obj_info["n_cls"]

        self.obj_idx = 6
        self.rel_idx = 10

        print(
            "Dataset (DataSetHyperSim) loaded: type {}, size {}".format(
                split, self.get_size()
            )
        )

    def _init_indexes(self, lvl_difficulty):

        if self.split == "train":
            file_name = "matching_frames.npy"
        else:
            file_name = "matching_frames_sample_obj.npy"
        ids_smpl_all = np.load(
            os.path.join(self.data_dir, file_name), allow_pickle=True
        )

        if self.split == "train":
            np.random.shuffle(ids_smpl_all)

        ids_smpl_all_addinfo = np.concatenate(
            [ids_smpl_all[:, 6:8], ids_smpl_all[:, 9:]], axis=1
        )
        ids_smpl_all = np.concatenate(
            [ids_smpl_all[:, :6], ids_smpl_all[:, 8:9]], axis=1
        ).astype(
            np.int32
        )  # add. info

        ids_sub = np.concatenate(
            [ids_smpl_all[ids_smpl_all[:, -1] == lvl] for lvl in lvl_difficulty], axis=0
        )  # difficult lvl wrt object measures
        ids_sub_addinfo = np.concatenate(
            [
                ids_smpl_all_addinfo[ids_smpl_all[:, -1] == lvl]
                for lvl in lvl_difficulty
            ],
            axis=0,
        )

        return ids_sub, ids_sub_addinfo

    def _count_unique_scene_ids(self):
        # manual size of dataset (ignore multiple pairs for same image)
        ids_unique = []
        tmp = np.zeros(
            (
                np.max(self.ids_sub[:, 0]) + 1,
                np.max(self.ids_sub[:, 1]) + 1,
                np.max(self.ids_sub[:, 2]) + 1,
            )
        )
        for smpl in self.ids_sub:
            if not tmp[int(smpl[0]), int(smpl[1]), int(smpl[2])]:
                tmp[int(smpl[0]), int(smpl[1]), int(smpl[2])] = 1.0
                ids_unique.append(smpl)
        return len(ids_unique)

    def set_bs(self, bs):
        self.bs = bs

    # --------------------------------
    # --- General functions:
    def get_size(self):
        """
        :return:    int, number of data samples
        """
        if self.split == "train":
            # return self.n_ids_unique
            return min(1000, self.n_ids_unique)
        else:
            return len(self.ids)

    def get_sample_names(self, input_batch):
        """

        :param input_batch:
        :return:
        """
        input_np = {}
        for k in ["scan_ids", "cam_ids", "frame_ids"]:
            if tf.is_tensor(input_batch[k]):
                input_np[k] = list(input_batch[k].numpy())
            else:
                input_np[k] = list(input_batch[k])

        n_views = len(input_np["scan_ids"])
        scene_names = [
            self.scene_names[input_np["scan_ids"][i]] for i in range(n_views)
        ]
        cam_names = [
            self.cam_names[input_np["scan_ids"][i]][input_np["cam_ids"][i]]
            for i in range(n_views)
        ]
        frame_names = [
            self.rgb_frame_names[input_np["scan_ids"][i]][input_np["cam_ids"][i]][
                input_np["frame_ids"][i]
            ]
            for i in range(n_views)
        ]

        return {
            "scene_names": scene_names,
            "cam_names": cam_names,
            "frame_names": frame_names,
        }

    # --------------------------------
    # --- Prepare input for model:
    def _prepare_viz_input(self, scene_rgb, idx, obj_2dbbs):
        """

        :param scene_rgb:       <PIL image>
        :param idx:             [<int>, <int>, <int>]
        :param obj_2dbbs:       (N_obj, 4)
        :return:    scene_rgb:      (H_scene, W_scene, 3)  -> <float>
                    obj_viz:        (N_obj, F) / (N_obj, H_obj, W_obj, 3)
        """

        scene_rgb = np.asarray(scene_rgb, dtype=np.float32) / 255.0

        if self.use_pretrained and self.obj_viz_feats is not None:
            scene_id, cam_id, frame_id = idx
            obj_viz = self.obj_viz_feats[scene_id][cam_id][frame_id]
            # -> some object features which got some invalid feature vector (holds for cropped features)
            obj_viz[obj_viz == -np.inf] = 0
            obj_viz = np.nan_to_num(obj_viz)
            if self.context_viz_feats is not None:
                n_obj = obj_viz.shape[0]
                context_viz = self.context_viz_feats[scene_id][cam_id][frame_id]
                context_viz = np.tile(np.expand_dims(context_viz, axis=0), (n_obj, 1))
                obj_viz = np.concatenate([obj_viz, context_viz], axis=-1)
        else:
            obj_rgb = [
                cv2.resize(scene_rgb[o[0] : o[2], o[1] : o[3]], self.img_shape_obj)
                for o in obj_2dbbs
            ]

            if obj_2dbbs.shape[0] > 0:
                obj_viz = np.stack(obj_rgb, axis=0)  # default case
            else:
                obj_viz = np.zeros(
                    (0, self.img_shape_obj[0], self.img_shape_obj[1], 3),
                    dtype=np.float32,
                )  # no obj.detections

        # -> smaller images, atm only used for visualization
        scene_rgb = cv2.resize(scene_rgb, (self.img_shape[1], self.img_shape[0]))

        return scene_rgb, obj_viz

    def _prepare_obj_class_input(self, idx):
        """

        :param idx:     [<int>, <int>, <int>]
        :return:        (N_obj, K)
        """
        scene_id, cam_id, frame_id = idx
        obj_lbl = np.asarray(
            self.obj_info["cls"][scene_id][cam_id][frame_id], np.int32
        )  # -1 offset in updated data already done
        obj_lbl_onehot = np.zeros((obj_lbl.size, self.n_classes), dtype=np.float32)
        obj_lbl_onehot[np.arange(obj_lbl.size), obj_lbl] = 1
        return obj_lbl_onehot

    def _prepare_obj_pose_input(self, idx, mpa, obj_2dbbs, bb_only=False):
        """

        :param idx:         [<int>, <int>, <int>]
        :param mpa:         <int>
        :param obj_2dbbs:   (N_obj, 4)
        :return:            (N_obj, D_obj=10)
        """

        scene_id, cam_id, frame_id = idx
        h_org, w_org = self.img_shape_org

        obj_bb_norm = np.asarray(
            [
                [o[0] / h_org, o[1] / w_org, o[2] / h_org, o[3] / w_org]
                for o in obj_2dbbs
            ],
            dtype=np.float32,
        )

        if bb_only:
            return obj_bb_norm

        obj_off = np.asarray(
            self.obj_info["offset"][scene_id][cam_id][frame_id], np.float32
        )  # cntr projection offset ( - normalization w.r.t. to bounding box/ object crop length already done)
        obj_dist = (
            np.expand_dims(
                np.asarray(
                    self.obj_info["dist"][scene_id][cam_id][frame_id], np.float32
                ),
                axis=-1,
            )
            * mpa
        )  # distance from camera camera, update depth->dist done
        obj_dist = (obj_dist - 0.5 * self.dist_factor) / self.dist_factor
        obj_cntr = (
            np.asarray(self.obj_info["cntr3d"][scene_id][cam_id][frame_id], np.float32)
            * mpa
        )

        obj_pose_info = np.concatenate(
            [obj_bb_norm, obj_off, obj_dist, obj_cntr], axis=-1
        ).astype(np.float32)

        return obj_pose_info

    def _prepare_rel_input(self, idx, mpa):
        """

        :param idx:         [<int>, <int>, <int>]
        :param mpa:         <int>
        :return:    rel_obj_idx:    (N_rel, 2)
                    rel_pose:       (N_rel,)
        """

        scene_id, cam_id, frame_id = idx
        obj_cntr = (
            np.asarray(self.obj_info["cntr3d"][scene_id][cam_id][frame_id], np.float32)
            * mpa
        )

        rel_obj_idx = np.reshape(
            np.asarray(
                [
                    [[i, j] for j in range(obj_cntr.shape[0])]
                    for i in range(obj_cntr.shape[0])
                ]
            ),
            (-1, 2),
        ).astype(np.int32)
        # rel_obj_idx = np.reshape(np.asarray([[[i, j] for j in range(obj_cntr.shape[0]) if i != j]
        #                                      for i in range(obj_cntr.shape[0])]), (-1, 2)).astype(np.int32)       # TODO: use this, as soon as we have flexible number of objects; otherwise sampling fails..

        rel_dist = obj_cntr[rel_obj_idx[:, 0]] - obj_cntr[rel_obj_idx[:, 1]]
        rel_dist = np.sqrt(np.sum(np.square(rel_dist), axis=-1))
        rel_pose = rel_dist  # currently only distance between objects

        return rel_obj_idx, rel_pose

    def prepare_single_frame(self, scene_id, cam_id, frame_id):

        scene_name = self.scene_names[scene_id]
        cam_name = self.cam_names[scene_id][cam_id]
        frame_name = self.rgb_frame_names[scene_id][cam_id][frame_id]
        idx = [scene_id, cam_id, frame_id]

        rgb_path = self.base_img_path.format(scene_name, cam_name, frame_name)
        scene_rgb = Image.open(rgb_path)

        cam_extr = self.camera_extr[scene_id][cam_id][frame_id].astype(np.float32)
        mpa = self.scene_info["meters_per_asset_unit"][scene_name]

        obj_id = np.asarray(self.obj_info["id"][scene_id][cam_id][frame_id], np.int32)

        obj_2dbbs = np.asarray(self.obj_info["bb2d"][scene_id][cam_id][frame_id])

        if self.split == "train" and not self.use_pretrained:
            if "bb" in self.augmentation_types:
                obj_2dbbs = self.augmentation_module.bb_noise(
                    obj_2dbbs, self.img_shape_org
                )

            if "rgb" in self.augmentation_types:
                scene_rgb = self.augmentation_module.enhance_rgb(scene_rgb)

        scene_rgb, obj_viz = self._prepare_viz_input(scene_rgb, idx, obj_2dbbs)

        # obj_subset_idx = range(obj_id.shape[0])  # is done later s.t. we have overlapping objects if possible

        if self.split == "train":  # additional data augmentation (object-level only)
            if "gauss" in self.augmentation_types:
                obj_viz = self.augmentation_module.gauss_noise(obj_viz, eps_type=0)

            if "homography_obj" in self.augmentation_types and not self.use_pretrained:
                obj_viz = self.augmentation_module.gauss_noise(
                    obj_viz, self.img_shape_obj, eps_type=0
                )

        obj_lbl_onehot = self._prepare_obj_class_input(idx)
        obj_pose = self._prepare_obj_pose_input(idx, mpa, obj_2dbbs)

        rel_obj_idx, rel_pose = self._prepare_rel_input(idx, mpa)

        assert frame_id.dtype == np.int32
        assert obj_id.dtype == np.int32
        assert (
            obj_viz.dtype == np.float32
        )  # obj_viz can be either obj_rgb or obj_viz_feats
        assert obj_lbl_onehot.dtype == np.float32
        assert rel_obj_idx.dtype == np.int32
        assert rel_pose.dtype == np.float32

        return (
            scene_id,
            cam_id,
            frame_id,
            scene_rgb,
            cam_extr,
            mpa,
            obj_id,
            obj_viz,
            obj_pose,
            obj_lbl_onehot,
            rel_obj_idx,
            rel_pose,
        )

    def _split_smpl_info(self, smpl, cur_id):
        """

        :param smpl:       (#data entries per frame)
        :param cur_id:      <int>
        :return:
        """
        smpl_scene = smpl[0 : self.obj_idx]
        smpl_obj = smpl[self.obj_idx : self.rel_idx] + (
            np.asarray([cur_id for _ in range(smpl[self.obj_idx].shape[0])]),
        )
        smpl_rel = smpl[self.rel_idx :] + (
            np.asarray([cur_id for _ in range(smpl[self.rel_idx].shape[0])]),
        )

        return smpl_scene, smpl_obj, smpl_rel

    def _get_sampled_obj_idx(self, smpl_1_obj, smpl_2_obj):
        """

        :param smpl_1/2_obj:  ((#obj data entries,) for each object)
        :param smpl_1/2_rel:  ((#rel data entries,) for each object)
        :return:    obj_subset_idx_1/2:     [<int> * #sampled_obj]
                    rel_subset_idx_1/2:     [<int> * (#sampled_obj * (#sampled_obj-1))]
        """

        # sub-sample objects from image pair
        #   - favors objects which occur in both images
        #   - if object count to small, re-sample some object
        intersect_obj_ids = list(set(smpl_1_obj[0]).intersection(smpl_2_obj[0]))
        if len(intersect_obj_ids) == self.n_smpl_obj:  # exactly the required number
            final_obj_ids_1 = intersect_obj_ids
            final_obj_ids_2 = final_obj_ids_1
        elif len(intersect_obj_ids) > self.n_smpl_obj:
            final_obj_ids_1 = random.sample(intersect_obj_ids, self.n_smpl_obj)
            final_obj_ids_2 = final_obj_ids_1
        else:

            def tmp_smpl(smpl, intersect_obj_ids):
                if self.n_smpl_obj <= smpl.shape[0]:  # enough objects in image
                    n_missing_obj = self.n_smpl_obj - len(intersect_obj_ids)
                    exclusive_obj_id = [i for i in smpl if i not in intersect_obj_ids]
                    obj_subset_idx = intersect_obj_ids + random.sample(
                        exclusive_obj_id, n_missing_obj
                    )
                else:
                    n_missing_obj = (
                        self.n_smpl_obj - smpl.shape[0]
                    )  # don't double-prioritize matchings, thus keep it simple
                    obj_subset_idx = list(smpl)
                    while n_missing_obj > len(smpl):
                        obj_subset_idx += list(smpl)
                        n_missing_obj = self.n_smpl_obj - len(obj_subset_idx)
                    obj_subset_idx += random.sample(list(smpl), n_missing_obj)
                return obj_subset_idx

            final_obj_ids_1 = tmp_smpl(smpl_1_obj[0], intersect_obj_ids)
            final_obj_ids_2 = tmp_smpl(smpl_2_obj[0], intersect_obj_ids)
        obj_subset_idx_1 = [list(smpl_1_obj[0]).index(o) for o in final_obj_ids_1]
        obj_subset_idx_2 = [list(smpl_2_obj[0]).index(o) for o in final_obj_ids_2]

        return obj_subset_idx_1, obj_subset_idx_2

    def _get_sampled_rel_idx(self, smpl_rel, obj_subset_idx):
        """

        :param smpl_rel:        ((#rel data entries,) for each object)
        :param obj_subset_idx:  [<int> for each object]
        :return:
        """

        # find valid relationships between sampled objects
        tmp = list(tuple(smpl_rel[0][i]) for i in range(smpl_rel[0].shape[0]))
        rel_subset_idx = [
            tmp.index(tuple([obj_subset_idx[r[0]], obj_subset_idx[r[1]]]))
            for r in list(product(range(self.n_smpl_obj), range(self.n_smpl_obj)))
            if r[0] != r[1]
        ]

        # get new object idx in [0, n_obj] wrt sampled object
        tmp_rel_ids = -1 * np.ones_like((smpl_rel[0]), dtype=np.int32)
        for i, n in enumerate(obj_subset_idx):
            tmp_rel_ids = np.where(smpl_rel[0] == n, i, tmp_rel_ids)
        smpl_rel = (tmp_rel_ids,) + smpl_rel[1:]

        return rel_subset_idx, smpl_rel

    def prepare_input(self, idx):

        scene_id = self.ids_sub[idx, 0]
        cam_id = self.ids_sub[idx, 1]
        frame_id = self.ids_sub[idx, 2]
        smpl_1 = self.prepare_single_frame(scene_id, cam_id, frame_id)
        smpl_1_scene, smpl_1_obj, smpl_1_rel = self._split_smpl_info(smpl_1, 0)

        cam_id = self.ids_sub[idx, 3]
        frame_id = self.ids_sub[idx, 4]
        smpl_2 = self.prepare_single_frame(
            scene_id, cam_id, frame_id
        )  # both new cam and frame id
        smpl_2_scene, smpl_2_obj, smpl_2_rel = self._split_smpl_info(smpl_2, 1)

        if self.split == "train":
            obj_subset_idx_1, obj_subset_idx_2 = self._get_sampled_obj_idx(
                smpl_1_obj, smpl_2_obj
            )
            rel_subset_idx_1, smpl_1_rel = self._get_sampled_rel_idx(
                smpl_1_rel, obj_subset_idx_1
            )
            rel_subset_idx_2, smpl_2_rel = self._get_sampled_rel_idx(
                smpl_2_rel, obj_subset_idx_2
            )
        else:
            obj_subset_idx_1 = range(len(smpl_1_obj[0]))
            obj_subset_idx_2 = range(len(smpl_2_obj[0]))
            rel_subset_idx_1 = range(len(smpl_1_rel[0]))
            rel_subset_idx_2 = range(len(smpl_2_rel[0]))

        res = []
        for i in range(len(smpl_1_scene)):  # entire scene info
            res.append(np.stack([smpl_1_scene[i], smpl_2_scene[i]], axis=0))
        for i in range(len(smpl_1_obj)):  # obj info (varying number)
            res.append(
                np.concatenate(
                    [smpl_1_obj[i][obj_subset_idx_1], smpl_2_obj[i][obj_subset_idx_2]],
                    axis=0,
                )
            )
        for i in range(len(smpl_1_rel)):  # rel info (varying number)
            res.append(
                np.concatenate(
                    [smpl_1_rel[i][rel_subset_idx_1], smpl_2_rel[i][rel_subset_idx_2]],
                    axis=0,
                )
            )
        return res

    def wrapped_generate_input(self):
        def f(idx):
            #     scene_id, cam_id, frame_id, scene_rgb, cam_extr, mpa,
            #     obj_id, obj_rgb/feat, obj_pose/bb, obj_lbl_onehot, obj_frame_id
            #     rel_idx, rel_pose, rel_frame_id
            return tf.py_function(
                func=self.prepare_input,
                inp=[idx],
                Tout=(
                    tf.int32,
                    tf.int32,
                    tf.int32,
                    tf.float32,
                    tf.float32,
                    tf.float32,
                    tf.int32,
                    tf.float32,
                    tf.float32,
                    tf.float32,
                    tf.int32,
                    tf.int32,
                    tf.float32,
                    tf.int32,
                ),
            )

        return f


class DataSetHyperSimFlex(DataSetHyperSim):

    def __init__(self, data_dir, split, cnfg):
        super(DataSetHyperSimFlex, self).__init__(data_dir, split, cnfg)

        self.max_n_smpl_obj = cnfg["max_n_smpl_obj"]

        self.obj_idx = 6
        self.rel_idx = 10

    def _get_sampled_idx(self, smpl_1_obj, smpl_2_obj, smpl_1_rel, smpl_2_rel):
        """
        sub-selection only for very high number of objects during training
        :return:
        """
        is_sampled = False
        if self.split == "train" and (
            smpl_1_obj[0].shape[0] > self.max_n_smpl_obj
            or smpl_2_obj[0].shape[0] > self.max_n_smpl_obj
        ):
            intersect_obj_ids = list(set(smpl_1_obj[0]).intersection(smpl_2_obj[0]))
            if (
                len(intersect_obj_ids) == self.max_n_smpl_obj
            ):  # exactly the required number
                final_obj_ids_1 = intersect_obj_ids
                final_obj_ids_2 = final_obj_ids_1
            elif (
                len(intersect_obj_ids) > self.max_n_smpl_obj
            ):  # more common objects than required
                final_obj_ids_1 = random.sample(intersect_obj_ids, self.max_n_smpl_obj)
                final_obj_ids_2 = final_obj_ids_1
            else:  # sampling of objects that only appear in one image required

                def tmp_smpl(smpl):
                    if smpl.shape[0] <= self.max_n_smpl_obj:
                        return smpl
                    else:
                        exclusive_obj_id = [
                            i for i in smpl if i not in intersect_obj_ids
                        ]
                        n_missing_obj = self.max_n_smpl_obj - len(intersect_obj_ids)
                        obj_subset_idx = intersect_obj_ids + random.sample(
                            exclusive_obj_id, n_missing_obj
                        )
                        return obj_subset_idx

                    # exclusive_obj_id = [i for i in smpl if i not in intersect_obj_ids]
                    # if self.n_smpl_obj <= smpl.shape[0]:    # enough objects in image
                    #     n_missing_obj = self.n_smpl_obj-len(intersect_obj_ids)
                    #     obj_subset_idx = intersect_obj_ids + random.sample(exclusive_obj_id, n_missing_obj)
                    # else:
                    #     n_missing_obj = self.n_smpl_obj-smpl.shape[0]  # don't double-prioritize matchings, thus keep it simple
                    #     obj_subset_idx = list(smpl)
                    #     while n_missing_obj > len(smpl):
                    #         obj_subset_idx += list(smpl)
                    #         n_missing_obj = self.n_smpl_obj- len(obj_subset_idx)
                    #     obj_subset_idx += random.sample(list(smpl), n_missing_obj)
                    # return obj_subset_idx

                final_obj_ids_1 = tmp_smpl(smpl_1_obj[0])
                final_obj_ids_2 = tmp_smpl(smpl_2_obj[0])
            obj_subset_idx_1 = [list(smpl_1_obj[0]).index(o) for o in final_obj_ids_1]
            obj_subset_idx_2 = [list(smpl_2_obj[0]).index(o) for o in final_obj_ids_2]

            # find valid relationships between sampled objects
            tmp_1 = list(tuple(smpl_1_rel[0][i]) for i in range(smpl_1_rel[0].shape[0]))
            tmp_2 = list(tuple(smpl_2_rel[0][i]) for i in range(smpl_2_rel[0].shape[0]))
            rel_subset_idx_1 = [
                tmp_1.index(tuple([obj_subset_idx_1[r[0]], obj_subset_idx_1[r[1]]]))
                for r in list(
                    product(range(len(obj_subset_idx_1)), range(len(obj_subset_idx_1)))
                )
                if r[0] != r[1]
            ]
            rel_subset_idx_2 = [
                tmp_2.index(tuple([obj_subset_idx_2[r[0]], obj_subset_idx_2[r[1]]]))
                for r in list(
                    product(range(len(obj_subset_idx_2)), range(len(obj_subset_idx_2)))
                )
                if r[0] != r[1]
            ]
            # # get new object idx in [0, n_obj] wrt sampled object
            # tmp_1_rel_ids = -1*np.ones_like((smpl_1_rel[0]), dtype=np.int32)
            # tmp_2_rel_ids = -1*np.ones_like((smpl_2_rel[0]), dtype=np.int32)
            # for j, n in enumerate(obj_subset_idx_1):
            #     tmp_1_rel_ids = np.where(smpl_1_rel[0] == n, j, tmp_1_rel_ids)
            # for j, n in enumerate(obj_subset_idx_2):
            #     tmp_2_rel_ids = np.where(smpl_2_rel[0] == n, j, tmp_2_rel_ids)
            # smpl_1_rel = (tmp_1_rel_ids,) + smpl_1_rel[1:]
            # smpl_2_rel = (tmp_2_rel_ids,) + smpl_2_rel[1:]
            is_sampled = True
        else:
            obj_subset_idx_1 = range(len(smpl_1_obj[0]))
            obj_subset_idx_2 = range(len(smpl_2_obj[0]))
            rel_subset_idx_1 = range(len(smpl_1_rel[0]))
            rel_subset_idx_2 = range(len(smpl_2_rel[0]))

        return (
            obj_subset_idx_1,
            obj_subset_idx_2,
            rel_subset_idx_1,
            rel_subset_idx_2,
            is_sampled,
        )

    @staticmethod
    def _update_rel_idx(rels, obj_subset_idxs):
        # get new object idx in [0, n_obj] wrt sampled object
        n_imgs = len(rels)
        tmp_rel_ids = [
            -1 * np.ones_like((rels[i][0]), dtype=np.int32) for i in range(n_imgs)
        ]
        for i in range(n_imgs):
            for j, n in enumerate(obj_subset_idxs[i]):
                tmp_rel_ids[i] = np.where(rels[i][0] == n, j, tmp_rel_ids[i])
        rels_new = [(tmp_rel_ids[i],) + rels[i][1:] for i in range(n_imgs)]
        return rels_new

    def prepare_input(self, idx_init):

        # TODO: is there a nicer way of sampling?
        if self.bs == 1:
            idx_batch = [idx_init.numpy()]
        else:
            # idx_batch = list(np.random.choice(self.ids, self.bs))
            idx_batch = [idx_init.numpy()] + list(
                np.random.choice(self.ids, self.bs - 1)
            )  # TODO: for some reason, this seems to be important

        all_smpls = []
        for i, idx in enumerate(idx_batch):
            scene_id = self.ids_sub[idx, 0]
            cam_id = self.ids_sub[idx, 1]
            frame_id = self.ids_sub[idx, 2]
            smpl_1 = self.prepare_single_frame(scene_id, cam_id, frame_id)
            smpl_1_scene, smpl_1_obj, smpl_1_rel = self._split_smpl_info(smpl_1, 0)

            cam_id = self.ids_sub[idx, 3]
            frame_id = self.ids_sub[idx, 4]
            smpl_2 = self.prepare_single_frame(
                scene_id, cam_id, frame_id
            )  # both new cam and frame id
            smpl_2_scene, smpl_2_obj, smpl_2_rel = self._split_smpl_info(smpl_2, 1)

            # sub-selection only for very high number of objects during training
            subset_idx = self._get_sampled_idx(
                smpl_1_obj, smpl_2_obj, smpl_1_rel, smpl_2_rel
            )
            (
                obj_subset_idx_1,
                obj_subset_idx_2,
                rel_subset_idx_1,
                rel_subset_idx_2,
                is_sampled,
            ) = subset_idx
            if is_sampled:
                smpl_1_rel, smpl_2_rel = self._update_rel_idx(
                    [smpl_1_rel, smpl_2_rel], [obj_subset_idx_1, obj_subset_idx_2]
                )

            cur_res = []
            for j in range(len(smpl_1_scene)):  # entire scene info
                cur_res.append(np.stack([smpl_1_scene[j], smpl_2_scene[j]], axis=0))
            n_fill = cur_res[-1].shape[0]
            cur_res.append(i * np.ones((n_fill,)))
            for j in range(len(smpl_1_obj)):  # obj info (varying number)
                if len(smpl_1_obj[j].shape) != 3:  # standard case
                    cur_res.append(
                        np.concatenate(
                            [
                                smpl_1_obj[j][obj_subset_idx_1],
                                smpl_2_obj[j][obj_subset_idx_2],
                            ],
                            axis=0,
                        )
                    )
                else:  # TODO: probably not needed anymore
                    cur_res.append(
                        np.stack([smpl_1_obj[j], smpl_1_obj[j]], axis=0)
                    )  # e.g. feature map (->image-wise)
            n_fill = cur_res[-1].shape[0]
            cur_res.append(i * np.ones((n_fill,)))
            for j in range(len(smpl_1_rel)):  # rel info (varying number)
                cur_res.append(
                    np.concatenate(
                        [
                            smpl_1_rel[j][rel_subset_idx_1],
                            smpl_2_rel[j][rel_subset_idx_2],
                        ],
                        axis=0,
                    )
                )
            n_fill = cur_res[-1].shape[0]
            cur_res.append(i * np.ones((n_fill,)))

            all_smpls.append(cur_res)

        res = []
        for j in range(len(all_smpls[0])):
            res.append(
                np.concatenate([all_smpls[i][j] for i in range(len(all_smpls))], axis=0)
            )
        return res

    def wrapped_generate_input(self):
        def f(idx):
            #     scene_id, cam_id, frame_id, scene_rgb, cam_extr, mpa, scene_split,
            #     obj_id, obj_rgb/feat, obj_pose/bb, obj_lbl_onehot, obj_frame_id, obj_split,
            #     rel_idx, rel_pose, rel_frame_id, rel_split
            return tf.py_function(
                func=self.prepare_input,
                inp=[idx],
                Tout=(
                    tf.int32,
                    tf.int32,
                    tf.int32,
                    tf.float32,
                    tf.float32,
                    tf.float32,
                    tf.int32,
                    tf.int32,
                    tf.float32,
                    tf.float32,
                    tf.float32,
                    tf.int32,
                    tf.int32,
                    tf.int32,
                    tf.float32,
                    tf.int32,
                    tf.int32,
                ),
            )

        return f


class DataSetHyperSimDirect(DataSetHyperSim):

    def __init__(self, data_dir, split, cnfg):
        super(DataSetHyperSimDirect, self).__init__(data_dir, split, cnfg)

        print(
            "[TMP] Dataset (DataSetHyperSim) -- for evaluating on training dataset "
            "(e.g. pre-computing viz feats or superglue matches)"
        )
        self.split = "test"

        # -> load image-pair indices
        self.ids_sub, _ = self._init_indexes(cnfg["lvl_difficulty"][split])
        if self.split == "train":
            self.n_ids_unique = self._count_unique_scene_ids()
        self.ids = np.arange(self.ids_sub.shape[0])

        print(
            "Dataset (DataSetHyperSim) loaded: type {}, size {}".format(
                split, self.get_size()
            )
        )

    def _init_indexes(self, lvl_difficulty):

        ids_full_all = np.load(
            os.path.join(self.data_dir, "matching_frames.npy"), allow_pickle=True
        )
        ids_full_all_first = ids_full_all[:, :3].astype(np.int32)
        ids_full_unique = []
        tmp = np.zeros(
            (
                np.max(ids_full_all_first[:, 0]) + 1,
                np.max(ids_full_all_first[:, 1]) + 1,
                np.max(ids_full_all_first[:, 2]) + 1,
            )
        )
        for smpl in ids_full_all:
            if not tmp[int(smpl[0]), int(smpl[1]), int(smpl[2])]:
                tmp[int(smpl[0]), int(smpl[1]), int(smpl[2])] = 1.0
                ids_full_unique.append(smpl)
        ids_smpl_all = np.stack(ids_full_unique, axis=0)

        ids_smpl_all_addinfo = np.concatenate(
            [ids_smpl_all[:, 6:8], ids_smpl_all[:, 9:]], axis=1
        )
        ids_smpl_all = np.concatenate(
            [ids_smpl_all[:, :6], ids_smpl_all[:, 8:9]], axis=1
        ).astype(
            np.int32
        )  # add. info

        return ids_smpl_all, ids_smpl_all_addinfo

    def get_size(self):
        """
        :return:    int, number of data samples
        """
        return super(DataSetHyperSimDirect, self).get_size()

    def _get_sampled_obj_idx(self, smpl_obj):
        """

        :param smpl_obj: ((#obj data entries,) for each object)
        :return:
        """
        obj_ids = smpl_obj[0]
        if len(obj_ids) == self.n_smpl_obj:  # exactly the required number
            final_obj_ids = obj_ids
        elif len(obj_ids) > self.n_smpl_obj:
            final_obj_ids = random.sample(list(obj_ids), self.n_smpl_obj)
        else:
            n_missing_obj = self.n_smpl_obj - obj_ids.shape[0]
            final_obj_ids = list(obj_ids)
            while n_missing_obj > len(obj_ids):
                final_obj_ids += list(obj_ids)
                n_missing_obj = self.n_smpl_obj - len(obj_ids)
            final_obj_ids += random.sample(list(obj_ids), n_missing_obj)
        obj_subset_idx = [list(smpl_obj[0]).index(o) for o in final_obj_ids]
        return obj_subset_idx

    def prepare_input(self, idx):

        scene_id = self.ids_sub[idx, 0]
        cam_id = self.ids_sub[idx, 1]
        frame_id = self.ids_sub[idx, 2]

        smpl = super(DataSetHyperSimDirect, self).prepare_single_frame(
            scene_id, cam_id, frame_id
        )
        smpl_scene = smpl[0 : self.obj_idx]
        smpl_obj = smpl[self.obj_idx : self.rel_idx]
        smpl_rel = smpl[self.rel_idx :]

        if self.split == "train":
            obj_subset_idx = self._get_sampled_obj_idx(smpl_obj)
            rel_subset_idx, smpl_rel = self._get_sampled_rel_idx(
                smpl_rel, obj_subset_idx
            )
        else:
            obj_subset_idx = range(len(smpl_obj[0]))
            rel_subset_idx = range(len(smpl_rel[0]))

        res = []
        for i in range(len(smpl_scene)):  # entire scene info
            res.append(smpl_scene[i])
        for i in range(len(smpl_obj)):  # obj/rel info (varying number)
            res.append(smpl_obj[i][obj_subset_idx])
        for i in range(len(smpl_rel)):  # obj/rel info (varying number)
            res.append(smpl_rel[i][rel_subset_idx])
        return res

    def wrapped_generate_input(self):
        def f(idx):
            #     scene_id, cam_id, frame_id, scene_rgb, cam_extr, mpa,
            #     obj_id, obj_rgb/feat, obj_pose/bb, obj_lbl_onehot
            #     rel_idx, rel_pose
            return tf.py_function(
                func=self.prepare_input,
                inp=[idx],
                Tout=(
                    tf.int32,
                    tf.int32,
                    tf.int32,
                    tf.float32,
                    tf.float32,
                    tf.float32,
                    tf.int32,
                    tf.float32,
                    tf.float32,
                    tf.float32,
                    tf.int32,
                    tf.float32,
                ),
            )

        return f


class DataSetHyperSimDetect(DataSetHyperSim):

    def __init__(self, data_dir, split, cnfg, detector):
        super(DataSetHyperSimDetect, self).__init__(data_dir, split, cnfg)

        if self.split == "train":
            print(
                "[ERROR] DataSetHyperSimFlexDetect.__init__(), dataset not yet configured for training."
            )
            exit(0)
        self.use_pretrained = False

        # self.obj_idx = 6
        self.rel_idx = 11

        self.detector_name = detector
        self.detection_dir = os.path.join(
            os.path.dirname(os.path.dirname(self.data_dir)),
            "Hypersim_Detections_" + self.detector_name,
        )
        detection_info = np.load(
            os.path.join(self.detection_dir, "assignment.npy"), allow_pickle=True
        )[()]
        if "subset" not in data_dir:
            self.detections = detection_info[split]
        else:
            self.detections = detection_info["train"]  # for debugging

        if "classes_intersect" in detection_info.keys():  # pre-trained on COCO only
            self.detections_classes = detection_info["classes_intersect"]["ids"][
                :-1
            ]  # TODO: with or without other property class?
        else:
            structure_cats = get_subset_categories(cnfg["cls_labeling"], "structure")
            self.detections_classes = [
                i for i in range(cnfg["num_cls_sem_obj"]) if i not in structure_cats
            ]

        self.detection_filters = ["class", "size"]

        self.model_type = None

    def set_model_type(self, model_type):
        """

        :param model_type:   <string>    ['matchVconst', 'matchVflex']
        :return:
        """
        if not (model_type in ["matchVconst", "matchVflex", "direct"]):
            print(
                f"[ERROR] DataSetHyperSimDetect.set_model_type(): Model type {model_type} not defined."
            )
            exit(0)
        self.model_type = model_type

    @staticmethod
    def _update_obj_ids(obj_ids_org, obj_ids_det, start_obj_id):
        """

        :param obj_ids_org:     (N_obj)
        :param obj_ids_det:     (N_obj)
        :param start_obj_id:    <int>
        :return:                (N_obj)
        """

        n_pred_obj = obj_ids_det.shape[0]

        missing_obj_id = np.arange(n_pred_obj) + start_obj_id
        obj_ids_det_updated = np.where(
            obj_ids_det >= 0, np.take(obj_ids_org, obj_ids_det), -missing_obj_id
        ).astype(np.int32)

        return obj_ids_det_updated

    def get_filter(self, obj_cls_detect, obj_2dbbs_detect):
        """

        :param obj_cls_detect:      (N_obj,)
        :param obj_2dbbs_detect:    (N_obj, 40)
        :return: filter:    (N_obj,)  -> <bool>
        """

        filter = [True for _ in obj_cls_detect]

        # if 'class_org' in self.detection_filters:  # only evaluate wrt. some objects in gt -> not used atm
        #     obj_cls_org = np.asarray(self.obj_info['cls'][scene_id][cam_id][frame_id], np.int32)
        #     cls_filter_org = [c in self.detections_classes for c in obj_cls_org]
        #
        #     # update 'gt'
        #     obj_cls_org = obj_cls_org[cls_filter_org]
        #     obj_id_org = np.asarray(self.obj_info['id'][scene_id][cam_id][frame_id], np.int32)[cls_filter_org]
        #     obj_2dbbs_org = np.asarray(self.obj_info['bb2d'][scene_id][cam_id][frame_id])[cls_filter_org]

        if "class" in self.detection_filters:
            cls_filter = [c in self.detections_classes for c in obj_cls_detect]
            filter = [filter[i] * cls_filter[i] for i in range(len(cls_filter))]
        if "size" in self.detection_filters:
            MIN_OBJ_SIZE = 25
            bb_filter = [
                (
                    bb2d[0] <= bb2d[2] - MIN_OBJ_SIZE
                    and bb2d[1] <= bb2d[3] - MIN_OBJ_SIZE
                )
                for bb2d in obj_2dbbs_detect
            ]
            filter = [bool(filter[i] * bb_filter[i]) for i in range(len(cls_filter))]

        return filter

    def prepare_single_frame(self, scene_id, cam_id, frame_id, start_obj_id):

        org_sample = super(DataSetHyperSimDetect, self).prepare_single_frame(
            scene_id, cam_id, frame_id
        )
        (
            scene_id,
            cam_id,
            frame_id,
            scene_rgb,
            cam_extr,
            mpa,
            obj_id_org,
            obj_viz,
            obj_pose_org,
            obj_lbl_onehot_org,
            rel_obj_idx,
            rel_pose,
        ) = org_sample

        scene_name = self.scene_names[scene_id]
        cam_name = self.cam_names[scene_id][cam_id]
        frame_name = self.rgb_frame_names[scene_id][cam_id][frame_id]
        idx = [scene_id, cam_id, frame_id]

        # load detections
        detects = self.detections[scene_name][cam_name][frame_name]
        obj_id_detect = np.asarray(detects["gt_obj_id"], np.int32)
        obj_id = self._update_obj_ids(obj_id_org, obj_id_detect, start_obj_id)
        obj_cls_detect = detects["pred_obj_cls"]
        obj_2dbbs_detect = np.asarray(detects["pred_obj_bb"])

        # filter detections
        filter = self.get_filter(obj_cls_detect, obj_2dbbs_detect)
        obj_id = obj_id[filter]
        obj_cls_detect = obj_cls_detect[filter]
        obj_2dbbs_detect = obj_2dbbs_detect[filter]

        n_obj_detect = obj_id.shape[0]

        # get rgb object crops wrt. detected bb (this update is required)
        rgb_path = self.base_img_path.format(scene_name, cam_name, frame_name)
        scene_rgb = Image.open(rgb_path)
        scene_rgb, obj_viz = self._prepare_viz_input(scene_rgb, idx, obj_2dbbs_detect)

        # get other object-wise properties
        if n_obj_detect > 0:
            obj_bb_norm = self._prepare_obj_pose_input(
                idx, mpa, obj_2dbbs_detect, bb_only=True
            )

            obj_lbl_onehot = np.zeros(
                (obj_cls_detect.size, self.n_classes), dtype=np.float32
            )
            obj_lbl_onehot[np.arange(obj_cls_detect.size), obj_cls_detect] = 1
        else:  # no valid detections for current image
            obj_bb_norm = np.zeros((0, 4), dtype=np.float32)

            obj_lbl_onehot = np.zeros((0, self.n_classes), dtype=np.float32)

        # get relationship indices
        rel_obj_idx = np.reshape(
            np.asarray(
                [
                    [[i, j] for j in range(obj_id.shape[0])]
                    for i in range(obj_id.shape[0])
                ]
            ),
            (-1, 2),
        ).astype(np.int32)

        assert frame_id.dtype == np.int32
        assert obj_id.dtype == np.int32
        assert obj_viz.dtype == np.float32
        assert obj_lbl_onehot.dtype == np.float32
        assert rel_obj_idx.dtype == np.int32

        # print(scene_id, frame_id, rel_dist_objidx.shape, rel_dist.shape, rel_sem_objidx.shape, rel_sem_onehot.shape)
        return (
            scene_id,
            cam_id,
            frame_id,
            scene_rgb,
            cam_extr,
            mpa,
            obj_id,
            obj_id_org,
            obj_viz,
            obj_bb_norm,
            obj_lbl_onehot,
            rel_obj_idx,
        )

    def prepare_input(self, idx):

        scene_id = self.ids_sub[idx, 0]
        cam_id = self.ids_sub[idx, 1]
        frame_id = self.ids_sub[idx, 2]
        smpl_1 = self.prepare_single_frame(scene_id, cam_id, frame_id, start_obj_id=1)
        smpl_1_scene, smpl_1_obj, smpl_1_rel = self._split_smpl_info(smpl_1, 0)

        cam_id = self.ids_sub[idx, 3]
        frame_id = self.ids_sub[idx, 4]
        smpl_2 = self.prepare_single_frame(
            scene_id, cam_id, frame_id, start_obj_id=1 + smpl_1_obj[0].shape[0]
        )  # both new cam and frame id
        smpl_2_scene, smpl_2_obj, smpl_2_rel = self._split_smpl_info(smpl_2, 1)

        if self.model_type == "matchVconst":
            res = []
            for i in range(len(smpl_1_scene)):  # entire scene info
                res.append(np.stack([smpl_1_scene[i], smpl_2_scene[i]], axis=0))
            for i in range(len(smpl_1_obj)):  # obj info (varying number)
                res.append(np.concatenate([smpl_1_obj[i], smpl_2_obj[i]], axis=0))
            for i in range(len(smpl_1_rel)):  # rel info (varying number)
                res.append(np.concatenate([smpl_1_rel[i], smpl_2_rel[i]], axis=0))
        elif self.model_type == "matchVflex":
            res = []
            for j in range(len(smpl_1_scene)):  # entire scene info
                res.append(np.stack([smpl_1_scene[j], smpl_2_scene[j]], axis=0))
            n_fill = res[-1].shape[0]
            res.append(0 * np.ones((n_fill,)))
            for j in range(len(smpl_1_obj)):  # obj info (varying number)
                if len(smpl_1_obj[j].shape) != 3:  # standard case
                    res.append(np.concatenate([smpl_1_obj[j], smpl_2_obj[j]], axis=0))
            n_fill = res[-1].shape[0]
            res.append(0 * np.ones((n_fill,)))
            for j in range(len(smpl_1_rel)):  # rel info (varying number)
                res.append(np.concatenate([smpl_1_rel[j], smpl_2_rel[j]], axis=0))
            n_fill = res[-1].shape[0]
            res.append(0 * np.ones((n_fill,)))
        else:
            print(
                f"[ERROR] DataSetHyperSimDetect.prepare_input(): Model type {self.model_type} is not valid or was not set."
            )
            exit(0)

        return res

    def wrapped_generate_input(self):
        def f_const(idx):
            #     scene_id, cam_id, frame_id, scene_rgb, cam_extr, mpa,
            #     obj_id, obj_id_org, obj_rgb/feat, obj_bb, obj_lbl_onehot, obj_frame_id,
            #     rel_idx, rel_frame_id
            return tf.py_function(
                func=self.prepare_input,
                inp=[idx],
                Tout=(
                    tf.int32,
                    tf.int32,
                    tf.int32,
                    tf.float32,
                    tf.float32,
                    tf.float32,
                    tf.int32,
                    tf.int32,
                    tf.float32,
                    tf.float32,
                    tf.float32,
                    tf.int32,
                    tf.int32,
                    tf.int32,
                ),
            )

        def f_flex(idx):
            #     scene_id, cam_id, frame_id, scene_rgb, cam_extr, mpa, scene_split,
            #     obj_id, obj_id_org, obj_rgb/feat, obj_pose/bb, obj_lbl_onehot, obj_frame_id, obj_split,
            #     rel_idx, rel_frame_id, rel_split
            return tf.py_function(
                func=self.prepare_input,
                inp=[idx],
                Tout=(
                    tf.int32,
                    tf.int32,
                    tf.int32,
                    tf.float32,
                    tf.float32,
                    tf.float32,
                    tf.int32,
                    tf.int32,
                    tf.int32,
                    tf.float32,
                    tf.float32,
                    tf.float32,
                    tf.int32,
                    tf.int32,
                    tf.int32,
                    tf.int32,
                    tf.int32,
                ),
            )

        if self.model_type == "matchVconst":
            return f_const
        else:
            assert self.model_type == "matchVflex"
            return f_flex


# ---------------------------------------------------------------------------------------------------------------------


def get_dataset(data_dir, model_ext, split, cnfg, detector="EfficientDet-D7"):
    if cnfg["name"] == "hypersim":
        if model_ext == "matchVconst":
            return DataSetHyperSim(data_dir, split, cnfg)
        elif model_ext == "matchVflex":
            return DataSetHyperSimFlex(data_dir, split, cnfg)
        elif model_ext == "direct":
            return DataSetHyperSimDirect(data_dir, split, cnfg)
        elif model_ext == "matchVdetect":
            return DataSetHyperSimDetect(data_dir, split, cnfg, detector=detector)
        else:
            print(
                f"[ERROR] DataLoader.get_dataset(): Model type {model_ext} not defined."
            )
            exit(1)
    else:
        print(
            "[ERROR] DataLoader.get_dataset(): Dataset {} not defined.".format(
                cnfg["name"]
            )
        )
        exit(1)
