"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

import tensorflow as tf
from tensorflow.keras import losses

cls_criterion = losses.CategoricalCrossentropy()
mse_criterion = losses.MeanSquaredError()
mae_criterion = losses.MeanAbsoluteError()


# ----------------------------------------
# GENERAL


def get_regr_loss(gt, pred):
    """

    :param gt:      (#entities, [D])
    :param pred:    (#entities, [D])
    :return:        (<float>)
    """
    return mse_criterion(gt, pred)


def get_regr_abs_loss(gt, pred):
    """

    :param gt:      (#entities, [D])
    :param pred:    (#entities, [D])
    :return:        (<float>)
    """
    return mae_criterion(gt, pred)


# ----------------------------------------
# CLASS + POSE


def get_cls_loss(cls_gt, cls_pred, lbl_weight):
    """

    :param cls_gt:          (BS, N_obj, K)
    :param cls_pred:        (BS*N_obj, K)
    :param lbl_weight:      (K,)
    :return:                (<float>)
    """
    n_classes = len(lbl_weight)
    cls_gt = tf.reshape(cls_gt, (-1, n_classes))
    cls_pred = tf.reshape(cls_pred, (-1, n_classes))

    lbl_weight = tf.expand_dims(lbl_weight, axis=0)
    lbl_weight = tf.reduce_sum(cls_gt * lbl_weight, axis=1)
    l_cls = cls_criterion(cls_gt, cls_pred, sample_weight=lbl_weight)
    return l_cls


def get_obj_pose_loss(pose_gt_dict, pose_pred_dict, add_info):
    """
    :param pose_gt_dict:  { e.g.
                            'obj_delta': (BS, N_obj, 2),
                            'obj_dist': (BS, N_obj, 1),
                            'obj_bb2d': (BS, N_obj, 4)}
    :param pose_pred_dict:  {>> respective keys as pose_gt_dict (relevant predictions)}
    :return:                (<float>)
    """

    for k in pose_gt_dict.keys():
        pose_gt_dict[k] = tf.reshape(pose_gt_dict[k], (-1, pose_gt_dict[k].shape[-1]))

    obj_filter, compensation_factor = None, 0
    losses = {}

    if "obj_delta_smplreg" in pose_pred_dict.keys():
        delta_gt = pose_gt_dict["obj_delta"]
        delta_pred = pose_pred_dict["obj_delta_smplreg"]

        # filter objects with cntr far outside image frame
        obj_filter_1 = tf.where(
            delta_gt <= 1.0, tf.ones_like(delta_gt), tf.zeros_like(delta_gt)
        )
        obj_filter_2 = tf.where(
            delta_gt >= -1.0, tf.ones_like(delta_gt), tf.zeros_like(delta_gt)
        )
        obj_filter = tf.math.reduce_prod(
            obj_filter_1 * obj_filter_2, axis=1, keepdims=True
        )
        n_val = tf.clip_by_value(
            tf.reduce_sum(obj_filter),
            1.0,
            tf.cast(tf.shape(delta_gt.shape)[0], tf.float32),
        )
        compensation_factor = tf.cast(tf.shape(delta_gt.shape)[0], tf.float32) / n_val

        # clip cntr-offset (this enables more stable training, however, yields potential bad final prediction)
        delta_pred = tf.clip_by_value(delta_pred, -1.0, 1.0)
        delta_gt = tf.clip_by_value(delta_gt, -1.0, 1.0)

        # l_delta = get_regr_loss(delta_gt, delta_pred)
        l_delta = compensation_factor * get_regr_loss(
            obj_filter * delta_gt, obj_filter * delta_pred
        )
        losses["l_delta"] = l_delta

        # control losses
        if "obj_bb2d" in pose_gt_dict.keys() and "img_shape" in add_info.keys():
            obj_bb2d = pose_gt_dict["obj_bb2d"]
            img_shape = add_info["img_shape"]
            obj_bb2d_length = tf.stack(
                [
                    (obj_bb2d[:, 2] - obj_bb2d[:, 0]) * img_shape[0],
                    (obj_bb2d[:, 3] - obj_bb2d[:, 1]) * img_shape[1],
                ],
                axis=1,
            )

            delta_pxl_gt = (
                delta_gt * obj_bb2d_length
            )  # BBs have different size over batch
            delta_pxl_pred = delta_pred * obj_bb2d_length

            l_delta_pxl_x = get_regr_abs_loss(delta_pxl_gt[:, 0], delta_pxl_pred[:, 0])
            l_delta_pxl_y = get_regr_abs_loss(delta_pxl_gt[:, 1], delta_pxl_pred[:, 1])
            losses["l_delta_pxl_x"] = l_delta_pxl_x
            losses["l_delta_pxl_y"] = l_delta_pxl_y

    if "obj_dist_smplreg" in pose_pred_dict.keys():
        dist_gt = pose_gt_dict["obj_dist"]
        dist_pred = pose_pred_dict["obj_dist_smplreg"]

        if obj_filter is None:
            l_dist = get_regr_loss(dist_pred, dist_gt)
        else:
            l_dist = compensation_factor * get_regr_loss(
                obj_filter * dist_gt, obj_filter * dist_pred
            )
        l_dist = tf.reduce_mean(l_dist)

        losses["l_dist"] = l_dist

    return losses


# ----------------------------------------
# SIMILARITY & AFFINITY


def get_sim_loss_cosine(obj_feat_sim_pair, obj_ids_match, alpha=0.5):
    """
    Cosine similarity.
    :param obj_feat_sim_pair:   [([BS,] N_obj1, F_sim), ([BS,] N_obj2, F_sim)]
    :param obj_ids_match:       ([BS,] N_obj1, N_obj2)
    :param alpha:               <float>
    :return:                    (<float>), (<float>), (<float>)
    """
    obj_feat_sim_norm_mat = tf.expand_dims(
        tf.linalg.norm(obj_feat_sim_pair[0], axis=-1), axis=-1
    ) * tf.expand_dims(tf.linalg.norm(obj_feat_sim_pair[1], axis=-1), axis=-2)

    sim_mat = tf.expand_dims(obj_feat_sim_pair[0], axis=-2) * tf.expand_dims(
        obj_feat_sim_pair[1], axis=-3
    )
    sim_mat = (
        tf.reduce_sum(sim_mat, axis=-1) / obj_feat_sim_norm_mat
    )  # (BS, N_obj1, N_obj2)
    sim_mat = 0.5 * (tf.ones_like(sim_mat) + sim_mat)

    n_common_obj = tf.reduce_sum(tf.cast(obj_ids_match, tf.float32))
    n_diff_obj = tf.reduce_sum(1 - tf.cast(obj_ids_match, tf.float32))

    sim_mat_pos = sim_mat[obj_ids_match]
    sim_mat_neg = sim_mat[tf.math.logical_not(obj_ids_match)]

    sim_mat_pos = tf.ones_like(sim_mat_pos) - sim_mat_pos
    sim_mat_neg = tf.clip_by_value(
        sim_mat_neg - alpha * tf.ones_like(sim_mat_neg), 0.0, 999.0
    )

    l_pos = tf.math.divide_no_nan(tf.reduce_sum(sim_mat_pos), n_common_obj)
    l_neg = tf.math.divide_no_nan(tf.reduce_sum(sim_mat_neg), n_diff_obj)

    n_smpls = tf.cast(tf.reduce_prod(tf.shape(obj_ids_match)), tf.float32)
    l_pos_2 = tf.math.divide_no_nan(tf.reduce_sum(sim_mat_pos), n_smpls)
    l_neg_2 = tf.math.divide_no_nan(tf.reduce_sum(sim_mat_neg), n_smpls)

    l_contrast = l_pos_2 + l_neg_2

    return l_contrast, l_pos, l_neg


def get_sim_loss_exp(obj_feat_sim_pair, obj_ids_match):
    """
    Exponential similarity.
    :param obj_feat_sim_pair:   [(BS, N_obj, F_sim), (BS, N_obj, F_sim)]
    :param obj_ids_match:       (BS, N_obj, N_obj)
    :return:                    (<float>), (<float>), (<float>)
    """
    sim_mat = tf.expand_dims(obj_feat_sim_pair[0], axis=2) - tf.expand_dims(
        obj_feat_sim_pair[1], axis=1
    )
    sim_mat = tf.exp(
        tf.math.sqrt(tf.reduce_sum(tf.math.square(sim_mat), axis=-1))
    )  # (BS, N_obj1, N_obj2)
    sim_mat = tf.ones_like(sim_mat) + sim_mat
    sim_mat = 2 * tf.ones_like(sim_mat) / sim_mat

    n_common_obj = tf.reduce_sum(tf.cast(obj_ids_match, tf.float32))
    n_diff_obj = tf.reduce_sum(1 - tf.cast(obj_ids_match, tf.float32))

    sim_mat_pos = sim_mat[obj_ids_match]
    sim_mat_neg = sim_mat[tf.math.logical_not(obj_ids_match)]

    sim_mat_pos = -tf.math.log(sim_mat_pos)
    sim_mat_neg = -tf.math.log(tf.ones_like(sim_mat_neg) - sim_mat_neg)

    l_pos = tf.math.divide_no_nan(tf.reduce_sum(sim_mat_pos), n_common_obj)
    l_neg = tf.math.divide_no_nan(tf.reduce_sum(sim_mat_neg), n_diff_obj)

    n_smpls = tf.cast(tf.reduce_prod(tf.shape(obj_ids_match)), tf.float32)
    l_pos_2 = tf.math.divide_no_nan(tf.reduce_sum(sim_mat_pos), n_smpls)
    l_neg_2 = tf.math.divide_no_nan(tf.reduce_sum(sim_mat_neg), n_smpls)

    l_contrast = l_pos_2 + l_neg_2

    return l_contrast, l_pos, l_neg


def get_sim_loss(obj_feat_sim_pair, obj_ids_match, loss_type):

    if loss_type == "cosine":
        return get_sim_loss_cosine(obj_feat_sim_pair, obj_ids_match)
    elif loss_type == "exp":
        return get_sim_loss_exp(obj_feat_sim_pair, obj_ids_match)
    else:
        print(f"[WARNING] losses.get_sim_loss(): loss_type={loss_type} not defined")


def get_affinity_loss(affinity_mat, obj_ids_match):
    """
    :param affinity_mat:        ((BS,) N_obj+1, N_obj+1)
    :param obj_ids_match:       ((BS,) N_obj, N_obj)
    :return:                    (<float>)
    """
    missing_obj_match_id1 = tf.math.logical_not(
        tf.math.reduce_any(obj_ids_match, axis=-2)
    )
    missing_obj_match_id2 = tf.math.logical_not(
        tf.math.reduce_any(obj_ids_match, axis=-1)
    )

    l_aff = (
        -tf.reduce_sum(affinity_mat[..., :-1, :-1][obj_ids_match])
        - tf.reduce_sum(affinity_mat[..., -1, :-1][missing_obj_match_id1])
        - tf.reduce_sum(affinity_mat[..., :-1, -1][missing_obj_match_id2])
    )

    l_aff = l_aff / tf.cast(tf.math.reduce_prod(tf.size(obj_ids_match)), tf.float32)

    return l_aff
