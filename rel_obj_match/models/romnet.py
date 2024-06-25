"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

import numpy as np

from utils.tf_funcs import weight_losses
from models.networks import *

from models.losses import *
from utils import tools_eval


class ROMNet(tf.keras.Model):

    def __init__(self, cnfg, name="romnet"):
        """
        Basic SceneGraphNetwork.
        :param cnfg:    model configuration dictionary
        """
        super(ROMNet, self).__init__(name=name)

        self.img_shape = cnfg["data"]["img_shape"]
        self.img_shape_obj = cnfg["data"]["img_shape_obj"]
        self.img_shape_org = cnfg["data"]["img_shape_org"]
        self.img_shape_coarse = cnfg["data"]["img_shape_coarse"]

        self.n_classes = cnfg["data"]["num_cls_sem_obj"]

        backbone_encoder_net_type = cnfg["model"]["backbone"]
        self.obj_encoder, self.preprocess_input_layer = load_encoder(
            backbone_encoder_net_type, self.img_shape_obj
        )

        if self.obj_encoder is not None:
            self.obj_encoder.trainable = False

    @staticmethod
    def get_input(batch_data):
        raise NotImplementedError("Subclass should implement get_input(..)")

    def call(self, inputs):
        raise NotImplementedError("Subclass should implement call(..)")

    def get_loss(self, output_data, gt_data, params):
        raise NotImplementedError("Subclass should implement get_loss(..)")

    def get_summary_data(self, loss_dict, is_training):
        """
        -> summary (over entire batch for losses etc., last batch for visualization)
        :param input_batch:     {...} see get_input()
        :param output:          {...} see call()
        :param losses:          {...} see get_loss()
        :param eval_dict:
        :param cnfg:
        :param is_training:     <bool>
        :param output_path:     <string>
        :return:
        """
        summary_data = {
            "inputs": {},
            "outputs": {},
            "losses": {},
            "eval": {},
            "params": {},
        }

        if is_training:
            for k, v in loss_dict.items():
                if k[0:2] == "l_" or k[0:6] == "l-ctrl" or k == "loss_total":
                    summary_data["losses"][k] = v
        else:
            summary_data["losses"]["loss_total"] = loss_dict["loss_total"]

        return summary_data

    def evaluate_batch(self, data, eval_dict, cnfg):
        raise NotImplementedError("Subclass should implement evaluate_batch(..)")


# ---------------------------------------------------------------------------------------------------------------------


class ROMNetMatch(ROMNet):
    def __init__(self, cnfg):
        super(ROMNetMatch, self).__init__(cnfg)

        self.dim_obj_global = self.n_classes
        self.dim_obj_cam = 2 + 1
        self.dim_obj_sim = cnfg["model"]["featnet_sim_layers"][-1]
        self.dim_rel_global = 1

        self.global_pooling_layer = tf.keras.layers.GlobalMaxPooling2D()

        self.obj_pos_encoder = create_mlp(
            cnfg["model"]["featnet_pose_enc_layers"], name="obj_feats"
        )

        # CAMERA VIEW DEPENDENT
        # -- simple mlp to compute camera view dependent features directly from viz+bb feats
        self.cam_feat_net = create_mlp(
            cnfg["model"]["featnet_prop_enc_layers"], name="cam_feats"
        )

        # CAMERA VIEW INDEPENDENT/ GLOBAL
        # -- simple mlp to compute camera view INdependent features directly from viz+bb feats
        self.global_feat_net = create_mlp(
            cnfg["model"]["featnet_prop_enc_layers"], name="cam_feats"
        )

        # AGNN
        # -- similarity features from attentional gnn
        self.obj_agnn = AttentionalGNN(self.dim_obj_sim, cnfg["model"]["agnn_layers"])

        # HEAD
        # -- prediction for cls, 3cntr (view), distance
        self.obj_global_head = create_mlp(
            [self.dim_obj_sim, self.dim_obj_global], name="obj_global_head"
        )
        self.obj_cam_head = create_mlp(
            [self.dim_obj_sim, self.dim_obj_cam], name="obj_cam_head"
        )
        self.obj_sim_head = create_mlp(
            [self.dim_obj_sim, self.dim_obj_sim], name="obj_sim_head"
        )
        self.rel_global_head = create_mlp(
            [self.dim_obj_sim, self.dim_rel_global], name="rel_head"
        )

        # args for sinkhorn algorithm
        self.bin_score = tf.Variable(1.0, name="aff_bin")
        self.sinkhorn_iters = cnfg["model"]["sinkhorn_iters"]

    @staticmethod
    def get_input(batch_data):

        return {
            "scan_ids": batch_data[0],  # (BS*2, 1)  --
            "cam_ids": batch_data[1],  # (BS*2, 1) --
            "frame_ids": batch_data[2],  # (BS*2, 1) --
            "scene_rgb": batch_data[3],  # (BS*2, W, H, 3)
            "cam_extr": batch_data[4],  # (BS*2, 4, 4)
            "mpa": batch_data[5],  # (BS*2,)
            "obj_ids": batch_data[
                6
            ],  # (BS, N_obj')    # -> variant-depended, (BS, N_obj) are merged -> BSxN_obj
            "obj_rgb": batch_data[
                7
            ],  # (BS, N_obj', W', H', 3) / (BS, N_obj', D_viz_obj)
            "obj_bb2d": batch_data[8][..., :4],  # (BS, N_obj', 4)
            "obj_delta2d": batch_data[8][..., 4:6],  # (BS, N_obj', 2)
            "obj_dist2d": batch_data[8][..., 6:7],  # (BS, N_obj', 1)
            "obj_cntr3d": batch_data[8][..., 7:],  # (BS, N_obj', 3)
            "obj_cls_gt": batch_data[9],  # (BS, N_obj', K_obj)
            "obj_frame_id": batch_data[10],  # (BS, N_obj')
            "rel_ids": batch_data[
                11
            ],  # (BS, N_rel', 2)   # -> variant-depended, (BS, N_rel) are merged -> BSxN_rel
            "rel_pose": batch_data[12],  # (BS, N_rel',)  # currently only distance
            "rel_frame_id": batch_data[13],  # (BS, N_rel')
        }

    # -- compute features from input
    def get_viz_features(self, rgb_obj_in):
        """
        :param rgb_obj_in:  (BS, N_obj', W', H', 3) / (BSxN_obj, W', H', 3)
        :return:            (BSxN_obj, F_viz)
        """

        # # Object-wise features
        if rgb_obj_in.shape[-1] != 3:  # already encoded feature
            obj_feat_viz = tf.reshape(
                rgb_obj_in, (-1, rgb_obj_in.shape[-1])
            )  # features already pre-computed
            return {
                "obj": obj_feat_viz,
            }

        if self.preprocess_input_layer is not None:
            rgb_obj_in = rgb_obj_in * 255
            rgb_obj_in = self.preprocess_input_layer(rgb_obj_in)

        rgb_obj_in = tf.reshape(
            rgb_obj_in, (-1, self.img_shape_obj[0], self.img_shape_obj[1], 3)
        )
        obj_feat_viz = self.obj_encoder(rgb_obj_in, training=False)
        obj_feat_viz = self.global_pooling_layer(obj_feat_viz)

        return {
            "obj": obj_feat_viz,
        }

    def get_pos_features(self, obj_2dbb):
        """
        :param obj_2dbb:    (BS, N_obj, 4) / (BSxN_obj, 4) -> [x_min, y_min, x_max, y_max]
        :return:            (BSxN_obj, F_pos)
        """

        obj_cntr2d = (
            tf.stack(
                [
                    0.5 * (obj_2dbb[..., 0] + obj_2dbb[..., 2]),
                    0.5 * (obj_2dbb[..., 1] + obj_2dbb[..., 3]),
                ],
                axis=-1,
            )
            - 0.5
        )
        obj_length = (
            tf.stack(
                [
                    (obj_2dbb[..., 2] - obj_2dbb[..., 0]),
                    (obj_2dbb[..., 3] - obj_2dbb[..., 1]),
                ],
                axis=-1,
            )
            - 0.5
        )
        obj_pos = tf.concat([obj_cntr2d, obj_length], axis=-1)
        obj_pos = tf.reshape(obj_pos, (-1, 4))

        pos_feat = self.obj_pos_encoder(obj_pos)
        return pos_feat

    # -- helpers: split/merge objects, relations etc.

    @staticmethod
    def _split_data(data, ids, n_splits):
        """
        :param data:        (BSxN_smpls, F)
        :param ids:         (BS, N_smpls) / (BS x N_smpls)
        :param n_splits:    <int>, e.g. n_batches, n_frames=2
        :return:
        """
        if len(ids.shape) == 2:  # batch_dim is used
            bs = ids.shape[0]
            data = tf.reshape(data, (bs, -1, data.shape[-1]))

        res = []
        for i in range(n_splits):
            smpl = data[ids == i]
            if len(ids.shape) == 2:
                smpl = tf.reshape(smpl, (bs, -1, data.shape[-1]))
            res.append(smpl)
        return res

    def _pairup_relwise_feats(self, feats_obj, rel_idx, rel_frame_id):
        """
        :param feats_obj:       [(BS, N_obj_1, F_obj), (BS, N_obj_2, F_obj)]   -> BS=1 for flexible number of objects
        :param rel_idx:         (BS, N_rel, 2)
        :param rel_frame_id:    (BS, N_rel)
        :return:                (BSxN_rel, F_rel)
        """

        dim_in = self.dim_obj_sim
        bs = tf.shape(feats_obj[0])[0]
        n_rel = tf.shape(rel_idx)[1]  # #rel over both frames

        tmp_batch_indices = tf.expand_dims(tf.range(bs), axis=-1)
        tmp_batch_indices = tf.tile(tmp_batch_indices, (1, n_rel))  # (BS, N_rel)

        rel_feat = []

        for fid in [0, 1]:
            feats_obj_cur_frame = feats_obj[fid]  # (BS, N_obj_i, F_sim)

            rel_idx_cur_frame = rel_idx[rel_frame_id == fid]  # (BS*N_rel_i, 2)

            tmp_batch_indices_cur_frame = tmp_batch_indices[
                rel_frame_id == fid
            ]  # (BS*N_rel_i, )

            subj_idx_cur_frame = tf.stack(
                [tmp_batch_indices_cur_frame, rel_idx_cur_frame[..., 0]], axis=-1
            )  # (BS*N_rel_i, 2)
            obj_idx_cur_frame = tf.stack(
                [tmp_batch_indices_cur_frame, rel_idx_cur_frame[..., 1]], axis=-1
            )

            subj_feat = tf.gather_nd(
                feats_obj_cur_frame, subj_idx_cur_frame
            )  # (BS*N_rel_i, F_obj)
            obj_feat = tf.gather_nd(feats_obj_cur_frame, obj_idx_cur_frame)
            rel_feat_cur_frame = tf.concat([subj_feat, obj_feat], axis=-1)

            rel_feat_cur_frame = tf.reshape(rel_feat_cur_frame, (bs, -1, 2 * dim_in))
            rel_feat.append(rel_feat_cur_frame)
        rel_feat = tf.concat(rel_feat, axis=1)  # (BS, N_rel, F_rel)
        rel_feat = tf.reshape(
            rel_feat, (-1, 2 * dim_in)
        )  # (BS=32/1*N_rel=2*20/?, 2*F_rel=256)

        return rel_feat

    # -- output predictions
    def get_obj_props(self, feats_cam, feats_global):
        """
        :param feats_cam:       (BSxN_obj, D_cam)
        :param feats_global:    (BSxN_obj, D_global)
        :return:                {   'obj_cls_pred':         (BSxN_obj, C),
                                    'obj_pose_delta_pred':  (BSxN_obj, 2),
                                    'obj_pose_dist_pred':   (BSxN_obj, 1)
                                }
        """

        if isinstance(feats_global, list):
            print("[NOT IMPLEMENTED] ROMNet.get_obj_props() - feats_global is list.")

        obj_out_cam = self.obj_cam_head(feats_cam)
        obj_out_global = self.obj_global_head(feats_global)

        obj_logits = obj_out_global[..., : self.n_classes]
        obj_cls = tf.nn.softmax(obj_logits)

        idx_delta = 0
        obj_pose_delta = obj_out_cam[..., idx_delta : idx_delta + 2]
        idx_dist = idx_delta + 2
        obj_pose_dist = obj_out_cam[..., idx_dist : idx_dist + 1]

        return {
            "obj_cls_pred": obj_cls,
            "obj_pose_delta_pred": obj_pose_delta,
            "obj_pose_dist_pred": obj_pose_dist,
        }

    def get_rel_props(self, feats_obj, rel_idx, rel_frame_id, rel_split_id):
        raise NotImplementedError("Subclass should implement get_rel_props(..)")

    def get_obj_sim_features(self, obj_feats, obj_ids, obj_frame_id):
        """

        :param obj_feats:       (BS*N_obj, F_in)    -> BS=1 for flexible number of objects
        :param obj_ids:         (BS, N_obj)
        :param obj_frame_id:    (BS, N_obj)
        :return:  {
                    'obj_feat_agnn': [(BS, N_obj_1, F_sim), (BS, N_obj_2, F_sim)],
                    'obj_feat_sim_pair': [(BS, N_obj_1, F_sim), (BS, N_obj_2, F_sim)]
                    'obj_ids_match': (BS, N_obj_1, N_obj_2)
                  }
        """

        obj_feat_0, obj_feat_1 = self._split_data(obj_feats, obj_frame_id, 2)

        obj_feat_agnn_pair = self.obj_agnn(obj_feat_0, obj_feat_1)
        obj_feat_agnn = tf.reshape(
            tf.concat(obj_feat_agnn_pair, axis=1), (-1, obj_feat_agnn_pair[0].shape[-1])
        )
        obj_feat_agnn_pair = self._split_data(obj_feat_agnn, obj_frame_id, 2)

        obj_feat_sim = self.obj_sim_head(obj_feat_agnn)
        obj_feat_sim_pair = self._split_data(obj_feat_sim, obj_frame_id, 2)

        # -> for later
        bs = obj_feat_0.shape[0]
        obj_ids_1 = tf.reshape(obj_ids[obj_frame_id == 0], (bs, -1, 1))
        obj_ids_2 = tf.reshape(obj_ids[obj_frame_id == 1], (bs, 1, -1))
        obj_ids_match = obj_ids_1 == obj_ids_2

        return {
            "obj_feat_agnn": obj_feat_agnn_pair,
            "obj_feat_sim": obj_feat_sim_pair,
            "obj_ids_match": obj_ids_match,
        }

    def get_obj_matching(self, obj_feat_sim_pair):
        """
        :param obj_feat_sim_pair:   [([BS,] N_obj_1, F_sim), ([BS,] N_obj_2, F_sim)]
        :return:    {
                        'obj_sim_mat': ([BS,] N_obj_1, N_obj_2),
                        'obj_affinity_mat': ([BS,] N_obj_1+1, N_obj_2+1)
                      }
        """

        n_dim = obj_feat_sim_pair[0].shape[-1]

        sim_mat = tf.expand_dims(obj_feat_sim_pair[0], axis=-2) * tf.expand_dims(
            obj_feat_sim_pair[1], axis=-3
        )
        sim_mat = tf.reduce_sum(sim_mat, axis=-1)  # ([BS,] N_obj1, N_obj2)
        sim_mat = sim_mat / n_dim**0.5  # -> from SuperGlue sim matrix

        # -> affinity matrix
        def_n_obj = None
        if hasattr(self, "n_smpl_obj"):
            def_n_obj = (self.n_smpl_obj, self.n_smpl_obj)
        aff_mat = sinkhorn(
            sim_mat, self.bin_score, iters=self.sinkhorn_iters, def_n_obj=def_n_obj
        )

        return {
            "obj_sim_mat": sim_mat,
            "obj_affinity_mat": aff_mat,
        }

    # -- training + evaluation
    def call(self, inputs):

        bs = tf.cast(inputs["scan_ids"].shape[0] / 2, tf.int32)

        feats_viz = self.get_viz_features(inputs["obj_rgb"])  # (BSxN_obj, D_viz)
        feats_pos = self.get_pos_features(inputs["obj_bb2d"])  # (BSxN_obj, D_pos)
        feats_obj = tf.concat([feats_viz["obj"], feats_pos], axis=-1)

        feats_cam = self.cam_feat_net(feats_obj)
        feats_global = self.global_feat_net(feats_obj)
        feats = tf.concat([feats_cam, feats_global], axis=-1)  # (BSxN_obj, D_..)

        # Object Matching
        if "obj_split" in inputs.keys():  # variable number of objects, split these up
            obj_sim_res = self.get_obj_sim_features(
                feats,
                inputs["obj_ids"],
                inputs["obj_frame_id"],
                inputs["obj_split"],
                bs,
            )
            obj_matching = self.get_obj_matching(obj_sim_res["obj_feat_sim"], bs)
        else:
            obj_sim_res = self.get_obj_sim_features(
                feats, inputs["obj_ids"], inputs["obj_frame_id"]
            )
            obj_matching = self.get_obj_matching(obj_sim_res["obj_feat_sim"])

        # Get object+relation properties
        obj_props = self.get_obj_props(feats_cam, feats_global)
        if "rel_split" in inputs.keys():  # variable number of relations, split these up
            rel_props = self.get_rel_props(
                obj_sim_res["obj_feat_agnn"],
                inputs["rel_ids"],
                inputs["rel_frame_id"],
                inputs["rel_split"],
            )
        else:
            rel_props = self.get_rel_props(
                obj_sim_res["obj_feat_agnn"], inputs["rel_ids"], inputs["rel_frame_id"]
            )

        res = {**obj_props, **rel_props, **obj_sim_res, **obj_matching}

        return res

    def get_loss(self, output_data, gt_data, params):

        with tf.name_scope("loss_obj_cls"):
            l_obj_cls = get_cls_loss(
                gt_data["obj_cls_gt"],
                output_data["obj_cls_pred"],
                params["obj_lbl_weight"],
            )

        with tf.name_scope("loss_obj_pos"):
            pose_gt_dict = {
                "obj_delta": gt_data["obj_delta2d"],
                "obj_dist": gt_data["obj_dist2d"],
                "obj_bb2d": gt_data[
                    "obj_bb2d"
                ],  # optional, both required for control loss (can be uncommented)
            }
            pose_pred_dict = {
                "obj_delta_smplreg": output_data["obj_pose_delta_pred"],
                "obj_dist_smplreg": output_data["obj_pose_dist_pred"],
            }
            add_info = {
                "img_shape": self.img_shape  # optional, both required for control loss (can be uncommented)
            }

            l_obj_pose_losses = get_obj_pose_loss(
                pose_gt_dict, pose_pred_dict, add_info
            )
            l_obj_pos = l_obj_pose_losses["l_delta"] + l_obj_pose_losses["l_dist"]

        with tf.name_scope("loss_rel_pose"):
            rel_pred = output_data["rel_pose_pred"]
            rel_gt = tf.reshape(gt_data["rel_pose"], tf.shape(rel_pred))
            l_rel_dist = get_regr_loss(rel_gt, rel_pred)

        with tf.name_scope("loss_similarity"):
            obj_feat_sim_pair = output_data["obj_feat_sim"]
            obj_ids_match = output_data["obj_ids_match"]
            l_contrast, l_pos, l_neg = 3 * [tf.zeros(())]
            if tf.is_tensor(obj_ids_match):
                l_contrast, l_pos, l_neg = get_sim_loss(
                    obj_feat_sim_pair, obj_ids_match, "cosine"
                )
            else:
                n_scenes = len(obj_ids_match)
                for bid in range(n_scenes):  # expect list over all scenes in batch
                    cur_l_contrast, cur_l_pos, cur_l_neg = get_sim_loss(
                        obj_feat_sim_pair[bid], obj_ids_match[bid], "cosine"
                    )
                    l_contrast += cur_l_contrast / n_scenes
                    l_pos += cur_l_pos / n_scenes
                    l_neg += cur_l_neg / n_scenes

        with tf.name_scope("loss_match"):
            l_aff = tf.zeros(())
            if tf.is_tensor(obj_ids_match):
                l_aff = get_affinity_loss(
                    output_data["obj_affinity_mat"], obj_ids_match
                )
            else:  # for flexible #obj
                n_scenes = len(output_data["obj_affinity_mat"])
                for bid in range(n_scenes):  # expect list over all scenes in batch
                    l_aff += (1 / n_scenes) * get_affinity_loss(
                        output_data["obj_affinity_mat"][bid], obj_ids_match[bid]
                    )

        loss_dict = {}
        loss_list = [
            ("obj-cls", l_obj_cls),
            ("obj-pos", l_obj_pos),
            ("rel-dist", l_rel_dist),
            ("contrastive", l_contrast),
            ("affinity", l_aff),
        ]
        loss = weight_losses(loss_list, loss_dict, params)

        loss_dict["loss_total"] = loss
        loss_dict["l-ctrl_contr_pos"] = l_pos
        loss_dict["l-ctrl_contr_neg"] = l_neg
        loss_dict["l-ctrl_pos_delta"] = l_obj_pose_losses["l_delta"]
        loss_dict["l-ctrl_pos_dist"] = l_obj_pose_losses["l_dist"]
        loss_dict["l-ctrl_pos_delta-x"] = l_obj_pose_losses["l_delta_pxl_x"]
        loss_dict["l-ctrl_pos_delta-y"] = l_obj_pose_losses["l_delta_pxl_y"]

        return loss_dict

    def get_summary_data(
        self,
        input_batch,
        output,
        loss_dict,
        eval_dict,
        cnfg,
        is_training,
        output_path=None,
    ):
        """
        -> summary (over entire batch for losses etc., last batch for visualization)
        :param input_batch:     {...} see get_input()
        :param output:          {...} see call()
        :param losses:          {...} see get_loss()
        :param eval_dict:
        :param cnfg:
        :param is_training:     <bool>
        :param output_path:     <string>
        :return:
        """
        summary_data = super(ROMNetMatch, self).get_summary_data(loss_dict, is_training)

        if is_training:
            # extract data
            idx = 0
            fid = 0
            fid_obj = (input_batch["obj_frame_id"][idx] == fid).numpy()
            scene_rgb_exmpl = np.reshape(
                input_batch["scene_rgb"],
                (-1, 2, self.img_shape[0], self.img_shape[1], 3),
            )[idx]
            obj_lbl_gt = np.argwhere(input_batch["obj_cls_gt"][idx][fid_obj] == 1)[:, 1]
            obj_bb = input_batch["obj_bb2d"][idx][fid_obj]

            # -> visualize pose estimation
            bs = input_batch["obj_ids"].shape[0]
            obj_lbl_pred = np.reshape(output["obj_cls_pred"], (bs, -1, self.n_classes))[
                idx
            ][fid_obj]
            obj_lbl_pred = np.argmax(obj_lbl_pred, axis=-1)

            obj_data_pos_list = [
                {
                    "obj_bb2d": obj_bb,
                    "obj_delta2d": input_batch["obj_delta2d"][idx][fid_obj],
                    "obj_dist": input_batch["obj_dist2d"][idx][fid_obj],
                    "obj_cls": obj_lbl_gt,
                },
                {
                    "obj_bb2d": obj_bb,
                    "obj_delta2d": np.reshape(
                        output["obj_pose_delta_pred"], (bs, -1, 2)
                    )[idx][fid_obj],
                    "obj_dist": np.reshape(output["obj_pose_dist_pred"], (bs, -1, 1))[
                        idx
                    ][fid_obj],
                    "obj_cls": obj_lbl_pred,
                },
            ]

            output_path_pos = output_path.replace("xxx", "pos")
            scene_rgb_exmpl_pose = tools_eval.viz_obj_pos(
                scene_rgb_exmpl[fid],
                obj_data_pos_list,
                cnfg["data"],
                output_path=output_path_pos,
            )
            summary_data["outputs"]["rgb_pos"] = tf.convert_to_tensor(
                scene_rgb_exmpl_pose
            )

            # -> visualize similarity/ matching
            fid_obj_other = (input_batch["obj_frame_id"][idx] == 1).numpy()
            obj_bb_2 = input_batch["obj_bb2d"][idx][fid_obj_other]
            obj_data_sim = {
                "obj_ids_match": output["obj_ids_match"][idx],  # (M, N), bool
                "obj_match_scores": output["obj_affinity_mat"][idx],
                "bb_list": [obj_bb, obj_bb_2],
            }
            output_path_sim = output_path.replace("xxx", "sim")
            scene_rgb_exmpl_sim = tools_eval.viz_obj_sim(
                scene_rgb_exmpl, obj_data_sim, output_path=output_path_sim
            )
            summary_data["outputs"]["rgb_sim"] = tf.convert_to_tensor(
                scene_rgb_exmpl_sim
            )

            # -> params
            summary_data["params"]["var_aff-bin"] = self.bin_score.numpy()

        return summary_data

    def evaluate_batch(self, data, eval_dict, cnfg):
        tools_eval.eval_quantitative_single_batch_class(
            data, eval_dict, ["obj_cls_gt", "obj_cls_pred"], cnfg
        )
        tools_eval.eval_quantitative_single_batch_pose(data, eval_dict, cnfg)
        tools_eval.eval_quantitative_single_batch_rel(data, eval_dict)
        tools_eval.eval_quantitative_single_batch_affinity(data, eval_dict)
        return eval_dict


class ROMNetMatchFlex(ROMNetMatch):

    def __init__(self, cnfg):
        super(ROMNetMatchFlex, self).__init__(cnfg)

    @staticmethod
    def get_input(batch_data):

        # -> remove pseudo batch size
        batch_data_new = []
        for i in range(len(batch_data)):
            batch_data_new.append(batch_data[i][0])
        batch_data = batch_data_new

        batch_data_base = batch_data[:6] + batch_data[7:12] + batch_data[13:16]
        parsed_batch = ROMNetMatch.get_input(batch_data_base)

        parsed_batch["scene_split"] = batch_data[6]
        parsed_batch["obj_split"] = batch_data[12]
        parsed_batch["rel_split"] = batch_data[16]

        return parsed_batch

    # -- output predictions
    def get_rel_props(self, feats_obj, rel_idx, rel_frame_id, rel_split_id):
        """

        :param feats_obj:       [[(N_obj_1, D_sim), (N_obj_2, D_sim)] for batch sample in BS],
        :param rel_idx:         (BSxN_rel)
        :param rel_frame_id:    (BSxN_rel, 2)
        :param rel_split_id:    (BSxN_rel)
        :return:                (BSxN_rel, D_rel)
        """

        bs = len(feats_obj)

        # prepare rel-wise input
        rel_idx = self._split_data(
            rel_idx, rel_split_id, bs
        )  # [(N_rel, 2) for batch sample in BS]
        rel_frame_id = self._split_data(
            rel_frame_id, rel_split_id, bs
        )  # [(N_rel,) for batch sample in BS]

        rel_feat_all = []
        for i in range(bs):
            tmp_feats_obj = [
                tf.expand_dims(feats_obj[i][0], axis=0),
                tf.expand_dims(feats_obj[i][1], axis=0),
            ]
            tmp_rel_idx = tf.expand_dims(rel_idx[i], axis=0)
            tmp_rel_frame_id = tf.expand_dims(rel_frame_id[i], axis=0)

            rel_feat_cur_smpl = super()._pairup_relwise_feats(
                tmp_feats_obj, tmp_rel_idx, tmp_rel_frame_id
            )
            rel_feat_all.append(rel_feat_cur_smpl)

        rel_feat = tf.concat(rel_feat_all, axis=0)
        rel_pose_pred = self.rel_global_head(rel_feat)

        return {
            "rel_pose_pred": rel_pose_pred,
        }

    def get_obj_sim_features(self, obj_feats, obj_ids, obj_frame_id, obj_split_id, bs):
        """

        :param obj_feats:       (BSxN_obj, F_obj)
        :param obj_ids:         (BSxN_obj)
        :param obj_frame_id:    (BSxN_obj)
        :param obj_split_id:    (BSxN_obj)
        :param bs:  <int>
        :return:  {
                    'obj_feat_agnn': [[(N_obj_1, F_sim), (N_obj_2, F_sim)] for batch sample in BS],
                    'obj_feat_sim_pair': [[(N_obj_1, F_sim), (N_obj_2, F_sim)] for batch sample in BS],
                    'obj_ids_match': [(N_obj_1, N_obj_2) for batch sample in BS]
                  }
        """

        obj_feats = self._split_data(obj_feats, obj_split_id, bs)
        obj_ids = self._split_data(obj_ids, obj_split_id, bs)
        obj_frame_id = self._split_data(obj_frame_id, obj_split_id, bs)

        obj_feat_agnn_pair_all = []
        obj_feat_sim_pair_all = []
        obj_ids_match_all = []
        for i in range(bs):
            res_cur_smpl = super().get_obj_sim_features(
                tf.expand_dims(obj_feats[i], axis=0),
                tf.expand_dims(obj_ids[i], axis=0),
                tf.expand_dims(obj_frame_id[i], axis=0),
            )

            # get ride of pseudo batch dim
            res_cur_smpl["obj_feat_agnn"] = [
                res_cur_smpl["obj_feat_agnn"][0][0],
                res_cur_smpl["obj_feat_agnn"][1][0],
            ]
            res_cur_smpl["obj_feat_sim"] = [
                res_cur_smpl["obj_feat_sim"][0][0],
                res_cur_smpl["obj_feat_sim"][1][0],
            ]
            res_cur_smpl["obj_ids_match"] = res_cur_smpl["obj_ids_match"][0]

            obj_feat_agnn_pair_all.append(res_cur_smpl["obj_feat_agnn"])
            obj_feat_sim_pair_all.append(res_cur_smpl["obj_feat_sim"])
            obj_ids_match_all.append(res_cur_smpl["obj_ids_match"])

        return {
            "obj_feat_agnn": obj_feat_agnn_pair_all,
            "obj_feat_sim": obj_feat_sim_pair_all,
            "obj_ids_match": obj_ids_match_all,
        }

    def get_obj_matching(self, obj_feat_sim_pair, bs):
        """

        :param obj_feat_sim_pair:   [[(N_obj_1, F_sim), (N_obj_2, F_sim)] for batch sample in BS]
        :param obj_ids:             (BSxN_obj)
        :param bs:                  <int>
        :return:    {
                    'obj_sim_mat': [(N_obj_1, N_obj_2) for batch sample in BS],
                    'obj_affinity_mat': [(N_obj_1, N_obj_2) for batch sample in BS]
                  }
        """

        sim_mat_all = []
        aff_mat_all = []
        for i in range(bs):
            res_cur_smpl = super().get_obj_matching(obj_feat_sim_pair[i])

            sim_mat_all.append(res_cur_smpl["obj_sim_mat"])
            aff_mat_all.append(res_cur_smpl["obj_affinity_mat"])

        return {
            "obj_sim_mat": sim_mat_all,
            "obj_affinity_mat": aff_mat_all,
        }

    # -- training
    def call(self, inputs):
        return super(ROMNetMatchFlex, self).call(inputs)

    def get_loss(self, output_data, gt_data, params):
        return super(ROMNetMatchFlex, self).get_loss(output_data, gt_data, params)

    def get_summary_data(
        self,
        input_batch,
        output,
        loss_dict,
        eval_dict,
        cnfg,
        is_training,
        output_path=None,
    ):

        # -> add pseudo batch size
        input_batch_new = {}
        for k, v in input_batch.items():
            if "obj_" in k or "rel_" in k:
                input_batch_new[k] = tf.expand_dims(v, axis=0)
            else:
                input_batch_new[k] = v

        return super(ROMNetMatchFlex, self).get_summary_data(
            input_batch_new,
            output,
            loss_dict,
            eval_dict,
            cnfg,
            is_training,
            output_path=output_path,
        )

    def evaluate_batch(self, data, eval_dict, cnfg):

        # -> add pseudo batch size
        data_new = {}
        for k, v in data.items():
            if tf.is_tensor(v) and ("obj_" in k or "rel_" in k) and not ("_pred" in k):
                data_new[k] = tf.expand_dims(v, axis=0)
            else:
                data_new[k] = v
        return super(ROMNetMatchFlex, self).evaluate_batch(data_new, eval_dict, cnfg)


class ROMNetMatchConst(ROMNetMatch):

    def __init__(self, cnfg):
        super(ROMNetMatchConst, self).__init__(cnfg)

        self.n_smpl_obj = cnfg["data"]["n_smpl_obj"]

    @staticmethod
    def get_input(batch_data):
        batch_data_new = []
        for b in batch_data:
            b_shape = b.shape
            if b_shape[1] == 2:
                if len(b_shape) == 2:
                    b = tf.reshape(b, [b_shape[0] * b_shape[1]])
                else:
                    b = tf.reshape(b, [b_shape[0] * b_shape[1]] + b_shape[2:])
            batch_data_new.append(b)
        batch_data = batch_data_new

        return ROMNetMatch.get_input(batch_data)

    # -- output predictions
    def get_rel_props(self, feats_obj, rel_idx, rel_frame_id):
        """

        :param feats_obj:       [(BS, N_obj_1, F_sim), (BS, N_obj_2, F_sim)]
        :param rel_idx:         (BS, N_rel)
        :param rel_frame_id:    (BS, N_rel, 2)
        :return:                (BS*N_rel, D_rel)
        """
        rel_feat = self._pairup_relwise_feats(feats_obj, rel_idx, rel_frame_id)

        rel_pose_pred = self.rel_global_head(rel_feat)

        return {
            "rel_pose_pred": rel_pose_pred,
        }

    # -- training
    def call(self, inputs):
        return super(ROMNetMatchConst, self).call(inputs)

    def get_loss(self, output_data, gt_data, params):
        return super(ROMNetMatchConst, self).get_loss(output_data, gt_data, params)


# ---------------------------------------------------------------------------------------------------------------------


class ROMNetVizFeats(ROMNet):

    def __init__(self, cnfg):
        super(ROMNetVizFeats, self).__init__(cnfg, name="VizFeatsNet")

        self.global_pooling_layer_max = tf.keras.layers.GlobalMaxPooling2D()
        self.global_pooling_layer_avg = tf.keras.layers.GlobalAveragePooling2D()

        backbone_encoder_net = cnfg["model"]["backbone"]
        if self.obj_encoder is None:
            print(
                "[ERROR] ROMNet.__init__(): Backbone Encoder Network {} unknown.".format(
                    backbone_encoder_net
                )
            )
            exit()

    @staticmethod
    def get_input(batch_data):
        # -> remove pseudo batch size (similar to flex variant - all objects required)
        scan_ids = batch_data[0]
        cam_ids = batch_data[1]
        frame_ids = batch_data[2]
        scene_rgb = batch_data[3]

        obj_id = batch_data[6]  # [0]

        obj_rgb = batch_data[7]  # [0]

        obj_bb2d = batch_data[8][..., :4]

        return {
            "scan_ids": scan_ids,  # (1, 1)  --
            "cam_ids": cam_ids,  # (1, 1) --
            "frame_ids": frame_ids,  # (1, 1) --
            "scene_rgb": scene_rgb,  # (1, W, H, 3)
            "obj_ids": obj_id,  # (N_obj)
            "obj_rgb": obj_rgb,  # (N_obj, W', H', 3)
            "obj_bb": obj_bb2d,  # (N_obj, 4)
        }

    def get_viz_features(self, rgb_scene_in, rgb_obj_in):
        """
        -> Compute obj feature map wrt crops on rgb image
        :param rgb_scene_in:   (BS*2, W', H', 3)
        :param rgb_obj_in:  (BS, N_obj', W', H', 3)
        :return:
        """

        # # Pre-processing
        if self.preprocess_input_layer is not None:
            rgb_scene_in = rgb_scene_in * 255  # cast back to 0-255 range
            rgb_scene_in = self.preprocess_input_layer(rgb_scene_in)

            rgb_obj_in = rgb_obj_in * 255
            rgb_obj_in = self.preprocess_input_layer(rgb_obj_in)

        # # Object feature
        rgb_obj_in = tf.reshape(
            rgb_obj_in, (-1, self.img_shape_obj[0], self.img_shape_obj[1], 3)
        )
        obj_featmap = self.obj_encoder(rgb_obj_in, training=False)
        obj_feat_max = self.global_pooling_layer_max(obj_featmap)
        obj_feat_avg = self.global_pooling_layer_avg(obj_featmap)

        return {
            "viz_feat_obj_max": obj_feat_max,
            "viz_feat_obj_avg": obj_feat_avg,
        }

    def get_viz_features_v2(self, rgb_scene_in, bb_obj_in):
        """
        -> Compute feature map wrt entire image, crop based on object bb
        :param rgb_scene_in:    (BS*2, W', H', 3)
        :param bb_obj_in:       (BS, N_obj', 4)
        :return:
        """

        # # Pre-processing
        if self.preprocess_input_layer is not None:
            rgb_scene_in = rgb_scene_in * 255  # cast back to 0-255 range
            rgb_scene_in = self.preprocess_input_layer(rgb_scene_in)

        # # Object feature
        img_featmap = self.img_encoder(
            rgb_scene_in, training=False
        )  # (BS, H'=8-24, W'=5-32, F1(=512..))

        roi_obj_feat = self.roi_pool([img_featmap, tf.expand_dims(bb_obj_in, axis=0)])[
            0
        ]

        obj_feat_max = self.global_pooling_layer_max(roi_obj_feat)

        roi_obj_feat_noninf = tf.where(
            tf.math.is_inf(roi_obj_feat), tf.zeros_like(roi_obj_feat), roi_obj_feat
        )
        roi_obj_feat_valid = tf.where(
            tf.math.is_inf(roi_obj_feat),
            tf.zeros_like(roi_obj_feat),
            tf.ones_like(roi_obj_feat),
        )
        roi_obj_feat_valid_n = tf.reduce_sum(
            tf.reduce_sum(tf.reduce_min(roi_obj_feat_valid, axis=-1), axis=2), axis=1
        )
        roi_obj_feat_sum = tf.reduce_sum(
            tf.reduce_sum(roi_obj_feat_noninf, axis=2), axis=1
        )
        obj_feat_avg = roi_obj_feat_sum / tf.expand_dims(roi_obj_feat_valid_n, axis=-1)

        return {
            "viz_feat_obj_max": obj_feat_max,
            "viz_feat_obj_avg": obj_feat_avg,
            "viz_featmap_scene": img_featmap,  # img_size /32 (2^5)
        }

    def call(self, inputs):

        return self.get_viz_features(inputs["scene_rgb"], inputs["obj_rgb"])


# ---------------------------------------------------------------------------------------------------------------------


def get_model(cnfg_model, ext):
    print("Load model: ROMNet-" + ext)
    if ext == "matchVconst":
        model = ROMNetMatchConst(cnfg_model)
    elif ext == "matchVflex":
        model = ROMNetMatchFlex(cnfg_model)
    elif ext == "vizfeat":
        model = ROMNetVizFeats(cnfg_model)
    else:
        print("ROMNet, Model Variation {} not defined.".format(ext))
        exit(1)

    return model
