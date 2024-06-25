"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import os

from utils import viz
from models.networks import sinkhorn_from_numpy


# --------------------------------------------------
# ---


def split_latent(latent, split):
    assert sum(split) == latent.shape[-1]
    res = []
    for i in range(len(split)):
        res.append(latent[..., sum(split[:i]) : sum(split[:i]) + split[i]])
    return res


# --------------------------------------------------
# ---  from network prediction to final representation


def get_2dcntr_from_repr(bb, delta, img_shape):
    """
    :param bb:          (N_obj(?), 4)
    :param delta:       (N_obj(?), 2)
    :param img_shape:   (2)
    :return:            (N_obj(?), 2)  <float>
    """

    bb_length_org = np.stack(
        [
            (bb[..., 2] - bb[..., 0]) * img_shape[0],
            (bb[..., 3] - bb[..., 1]) * img_shape[1],
        ],
        axis=1,
    )
    bb_cntr = np.stack(
        [
            (bb[..., 0] + (bb[..., 2] - bb[..., 0]) / 2.0) * img_shape[0],
            (bb[..., 1] + (bb[..., 3] - bb[..., 1]) / 2.0) * img_shape[1],
        ],
        axis=-1,
    )

    delta_pxl = np.stack(
        [delta[:, 0] * bb_length_org[:, 0], delta[:, 1] * bb_length_org[:, 1]], axis=-1
    )

    return bb_cntr + delta_pxl


def get_3dcntr_from_repr(cntr2d, dist, cam_K, cam_R=None):
    """
    cntr2d:     (N_obj(?), 2)
    dist:       (N_obj(?), 1)
    cam_K:      (3, 3)
    cam_R:      (4, 4)
    :return:    (N_obj(?), 3)
    """

    shape_org = cntr2d.shape[:-1]
    cntr2d = np.reshape(cntr2d, (-1, 2, 1))
    dist = np.reshape(dist, (-1, 1, 1))

    cntr2d = np.stack(
        [cntr2d[:, 1, :], cntr2d[:, 0, :], np.ones_like(cntr2d[:, 0, :])], axis=1
    )  # (-1, 3, 1)  # change to image-coords necessary! (this works now)
    cam_K_inv = np.expand_dims(np.linalg.inv(cam_K), axis=0)

    cntr3d = np.matmul(cam_K_inv, cntr2d)
    dist_to_depth_ratio = np.sqrt(np.sum(np.square(cntr3d), axis=1, keepdims=True))
    cntr3d = (dist / dist_to_depth_ratio) * cntr3d

    # transform to toward-up-right coordinate system (doing some reasoning, this changing of axes makes sense)
    cntr3d = np.stack(
        [
            cntr3d[:, 0, :],
            -cntr3d[:, 1, :],
            -cntr3d[:, 2, :],
            np.ones_like(cntr3d[:, 0, :]),
        ],
        axis=1,
    )

    if cam_R is not None:
        cam_R_inv = np.expand_dims(np.linalg.inv(cam_R), axis=0)
        cntr3d = cam_R_inv @ cntr3d
    cntr3d = cntr3d[:, :3, :] / cntr3d[:, 3:4, :]

    cntr3d = np.reshape(cntr3d, shape_org + (3,))

    return cntr3d


def get_3dbb_from_repr(
    cntr3d, size, ori, sem_cls, bins_size_avg=None, cam_R=None, bb_with_ori=False
):
    """
    :param cntr3d:          (N_obj, 3)
    :param size:            (N_obj, 3)
    :param ori:             (N_obj, )
    :param sem_cls:         (N_obj, )
    :param bins_size_avg:   (N_cls, 3)
    :return:                (N_obj, 8, 3)
    """

    coeffs = np.exp(size)
    centroid = cntr3d

    al = np.reshape(coeffs, (-1, 1, 3, 1))
    c = np.reshape(centroid, (-1, 1, 3, 1))

    ori2 = np.expand_dims(
        np.expand_dims(
            np.stack([np.cos(ori), np.zeros_like(ori), np.sin(ori)], axis=-1), axis=1
        ),
        axis=-1,
    )

    if cam_R is not None:
        cam_R = np.reshape(cam_R, (1, 1, 4, 4))
        cam_R_inv = np.linalg.inv(cam_R)
        na = cam_R_inv[:, :, :3, :3] @ ori2
        na2 = np.stack(
            [
                na[:, :, 0, 0] / np.sqrt(na[:, :, 0, 0] ** 2 + na[:, :, 1, 0] ** 2),
                na[:, :, 1, 0] / np.sqrt(na[:, :, 0, 0] ** 2 + na[:, :, 1, 0] ** 2),
            ],
            axis=-1,
        )
    na3 = np.stack(
        [np.arccos(na2[:, :, 0]), np.arcsin(na2[:, :, 1]), np.zeros_like(na2[:, :, 1])],
        axis=-1,
    )
    na4 = na3[:, :, 0]
    na4b = na3[:, :, 1]
    na_final = np.stack(
        [
            np.stack([np.cos(na4), -np.sin(na4), np.zeros_like(na4)], axis=-1),
            np.stack([np.sin(na4), np.cos(na4), np.zeros_like(na4)], axis=-1),
            np.stack(
                [np.zeros_like(na4), np.zeros_like(na4), np.ones_like(na4)], axis=-1
            ),
        ],
        axis=-2,
    )
    na_final = np.reshape(na_final, (-1, 1, 3, 3))

    bb3d = (
        np.asarray(
            [
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                ]
            ]
        )
        - 0.5
    )
    if bb_with_ori:
        ori = np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        bb3d = np.concatenate([bb3d, ori], axis=1)

    bb3d = np.expand_dims(bb3d, axis=-1)

    bb3d = na_final @ (al * bb3d) + c

    return bb3d


def viz_obj_pos(scene_rgb, obj_data_list, cnfg, output_path=None):
    """

    :param scene_rgb:       (H, W, 3)
    :param obj_data_list:   [{..}], see below
    :param cnfg:            data_cnfig, i.e. {'dist_factor':..., ...}
    :param output_path:     <string> (opt
    :return:
    """

    scene_rgb_pose_all = []
    for data in obj_data_list:  # iterate over several images from single scene
        obj_bb2d = data["obj_bb2d"]  # (N_obj, 2)
        obj_delta = data["obj_delta2d"]  # (N_obj, 2)
        obj_dist = data["obj_dist"]  # (N_obj, 1)
        obj_cls = data["obj_cls"]  # (N_obj, K)

        cntr_proj = get_2dcntr_from_repr(obj_bb2d, obj_delta, scene_rgb.shape[0:2])
        obj_dist_mtrs = obj_dist * cnfg["dist_factor"] + 0.5 * cnfg["dist_factor"]

        scene_rgb_pose = viz.draw_bb(np.copy(scene_rgb), obj_bb2d, sem_lbls=obj_cls)
        scene_rgb_pose = viz.draw_cntrs(
            scene_rgb_pose, cntr_proj, bb=obj_bb2d, dist=obj_dist_mtrs, sem_lbls=obj_cls
        )
        scene_rgb_pose_all.extend([scene_rgb_pose, np.ones((scene_rgb.shape[0], 5, 3))])

    scene_rgb_pose_all = np.concatenate(scene_rgb_pose_all[:-1], axis=1)

    if output_path is not None:
        viz.show_image(scene_rgb_pose_all, path=output_path)

    return scene_rgb_pose_all


def viz_obj_sim(scene_rgb, obj_data, output_path=None):
    """

    :param scene_rgb:       (2, H, W, 3)
    :param obj_data_list:   {..}, see below
    :param cnfg:            data_cnfig, i.e. {'dist_factor':..., ...}
    :param output_path:     <string> (opt
    :return:
    """

    scene_rgb_sim = np.copy(scene_rgb)
    obj_match_scores = np.exp(
        obj_data["obj_match_scores"]
    )  # predicted output in log format

    scene_rgb_sim = viz.draw_sim(
        scene_rgb_sim,
        obj_data["obj_ids_match"],  # (M, N), <bool>
        obj_match_scores,  # (M, N), <float>
        obj_data["bb_list"],
    )  # [(4,), (4,)]

    if output_path is not None:
        viz.show_image(scene_rgb_sim, path=output_path)

    return scene_rgb_sim


# --------------------------------------------------
# ---


def class_accuracy(cls_gt_onehot, cls_pred_prob, sum_samples=True):
    """
    Compute accuracy for classification task (overall mean, per-class).
    :param cls_gt_onehot:   (N_obj, K)
    :param cls_pred_prob:   (N_obj, K)
    :param sum_samples:     <bool   [default=True]
    :return:    cls_acc:        <int>   # actually correspond to count of correct objects; averaging is done at the end
                cls_acc_pc:     (K,)
                n:              <int>   -> count of entities
                cls_gt, cls_pred:   (N_obj,)    -> class label
    """

    cls_gt = np.argmax(cls_gt_onehot, axis=1)
    cls_pred = np.argmax(cls_pred_prob, axis=1)
    n = cls_gt.shape[
        0
    ]  # mixes obj./ rel. over all images in batch (-> single image only during evaluation)

    cls_acc = cls_gt == cls_pred

    cls_pred_onehot = np.zeros_like(cls_pred_prob, dtype=np.float32)
    cls_pred_onehot[np.arange(n), np.argmax(cls_pred_prob, axis=1)] = 1
    cls_acc_pc = cls_gt_onehot * cls_pred_onehot

    if sum_samples:
        cls_acc = np.sum(cls_acc)  # sum over samples
        cls_acc_pc = np.sum(cls_acc_pc, axis=0)

    return cls_acc, cls_acc_pc, cls_gt, cls_pred


def position_error(
    obj_data_gt,
    obj_data_pred,
    obj_bb2d,
    mpa,
    dist_factor,
    img_shape,
    filter_invalid_obj=True,
):
    """

    :param obj_data_gt:     [(N_obj, 2), (N_obj, 1)]
    :param obj_data_gt:     [(N_obj, 2), (N_obj, 1)]
    :param obj_bb2d:        (N_obj, 4)
    :param mpa:             <int>
    :param dist_factor:     <int>
    :param img_shape:       (2,) - <int>
    :param filter_invalid_obj   <bool>
    :return:    dist_cntr3d     (N_obj,)    (distance between re-projected object center points)
                n_val           <int>       (count of objects with center projected in image)
    """

    cam_K = np.asarray(
        [[886.81001348, 0.0, 512.0], [0.0, 886.81001348, 384.0], [0.0, 0.0, 1.0]]
    )

    obj_delta_gt, obj_dist_gt = obj_data_gt
    obj_delta_pred, obj_dist_pred = obj_data_pred

    # count how many objects are 'valid', i.e. their center is projected (close to) image plane
    obj_filter = np.asarray((obj_delta_gt <= 1), dtype=np.float32) * np.asarray(
        (obj_delta_gt >= -1), dtype=np.float32
    )
    obj_filter = np.prod(obj_filter, axis=1, dtype=bool)
    n_val = np.sum(obj_filter)

    cntr2d_gt = get_2dcntr_from_repr(obj_bb2d, obj_delta_gt, img_shape)
    cntr2d_pred = get_2dcntr_from_repr(obj_bb2d, obj_delta_pred, img_shape)

    obj_dist_gt = (obj_dist_gt * dist_factor + 0.5 * dist_factor) / mpa
    obj_dist_pred = (obj_dist_pred * dist_factor + 0.5 * dist_factor) / mpa

    cntr3d_gt = (
        get_3dcntr_from_repr(cntr2d_gt, obj_dist_gt, cam_K) * mpa
    )  # object position in camera-view
    cntr3d_pred = get_3dcntr_from_repr(cntr2d_pred, obj_dist_pred, cam_K) * mpa

    transl_cntr3d = cntr3d_pred - cntr3d_gt
    if filter_invalid_obj:
        transl_cntr3d = transl_cntr3d[
            obj_filter
        ]  # this makes more sense wrt trainings pipeline (+ also fits description)
    dist_cntr3d = np.sqrt(np.sum(transl_cntr3d * transl_cntr3d, axis=1))

    return dist_cntr3d, n_val


# --------------------------------------------------
# ---


def sinkhorn(sim_mat, alpha, iters=10):
    return sinkhorn_from_numpy(sim_mat, alpha, iters)


# --------------------------------------------------
# ---


def init_affinity_entries(
    eval_dict, aff_type, name="", bins_info=None, array_info=None
):
    """

    :param eval_dict:   {}
    :param aff_type:    <string>
    :param name:        <string> [e.g. 'scenewise', ...]
    :param bins_info:   [keys: [<string>], keys: [[<float>,]]]
    :param array_info:  [shape: (<int>)]
    :return:
    """
    if aff_type == "basic":
        for t in ["n_match_gt", "n_missing_gt"]:
            eval_dict[f"{t}"] = 0
        for t in ["n-matches", "n-missing", "n-tp", "n-fp", "n-tn", "n-fn"]:
            eval_dict[f"matching-S_{t}"] = 0
    elif aff_type == "dict":
        assert name != ""
        for t in [
            "n-matches",
            "n-missing",
            "n-tp",
            "n-fp",
            "n-tn",
            "n-fn",
            "prec",
            "rec",
            "acc",
        ]:
            eval_dict[f"matching-D_{name}_{t}"] = {}
    elif aff_type == "bin":
        assert name != ""
        assert bins_info is not None
        bins_keys, bins_values = bins_info
        bins_shape = tuple([len(v) for v in bins_values])
        for t in ["n-matches", "n-missing", "n-tp", "n-fp", "n-tn", "n-fn"]:
            eval_dict[f"matching-B_{name}_{t}"] = np.zeros(bins_shape)
        eval_dict["info_matching_bins"] = [bins_keys, bins_values]
    elif aff_type == "array":
        assert name != ""
        assert array_info is not None
        shape = array_info[0]
        for t in ["n-matches", "n-missing", "n-tp", "n-fp", "n-tn", "n-fn"]:
            eval_dict[f"matching-A_{name}_{t}"] = np.zeros(shape)
    else:
        print(
            f"[WARNING] tools_eval.init_affinity_entries(), aff_type={aff_type} is unknown."
        )


def structure_gt_data(obj_ids_match):
    """

    :param obj_ids_match:       (M, N)  - tf.bool
    :return:
    """
    if not isinstance(obj_ids_match, np.ndarray):  # tf.Tensor
        obj_ids_match = obj_ids_match.numpy().astype(np.int32)
    n_obj1, n_obj2 = obj_ids_match.shape

    obj_ids_missing_m = np.ones((n_obj1,)) - np.clip(
        np.sum(obj_ids_match, axis=1), 0, 1
    )  # (M,)  # -> diffuculty about object detections (multiple objects assigned to same gt object
    obj_ids_missing_n = np.ones((n_obj2,)) - np.clip(
        np.sum(obj_ids_match, axis=0), 0, 1
    )  # (N,)

    total_gt_matches = np.sum(obj_ids_match)
    total_gt_missing = np.sum(obj_ids_missing_m) + np.sum(obj_ids_missing_n)

    return (
        obj_ids_match,
        [n_obj1, n_obj2],
        [obj_ids_missing_m, obj_ids_missing_n],
        [total_gt_matches, total_gt_missing],
    )


def max_pred_assignment(affinity_mat_ext):
    """

    :param affinity_mat_ext:    (M+1, N+1)
    :return: aff_matrix_list:   [(M, N), (M, N)]
             dustbin_list:      [(M,), (N,) ]
    """

    def max_assignment(mat, axis):
        mat_hard = np.zeros_like(mat)
        mat_max_ind = np.argmax(mat, axis=axis)
        mat_max_ind_other = np.arange(mat_max_ind.shape[0])
        if axis == 0:
            mat_max_ind = np.stack([mat_max_ind, mat_max_ind_other], axis=-1)
        else:  # axis = 1
            mat_max_ind = np.stack([mat_max_ind_other, mat_max_ind], axis=-1)

        for i in range(mat_max_ind.shape[0]):
            mat_hard[mat_max_ind[i, 0], mat_max_ind[i, 1]] = 1.0
        return mat_hard

    affinity_mat_max_m = max_assignment(affinity_mat_ext, axis=1)
    affinity_mat_max_n = max_assignment(affinity_mat_ext, axis=0)

    aff_matrix_list = [affinity_mat_max_m[:-1, :-1], affinity_mat_max_n[:-1, :-1]]
    dustbin_list = [affinity_mat_max_m[:-1, -1], affinity_mat_max_n[-1, :-1]]

    return aff_matrix_list, dustbin_list


def count_correct_matches(aff_mat, db, match_gt, missing_gt):
    """

    -> N_obj_2 can be NULL, e.g. for object-wise evaluation
    :param aff_mat:     (N_obj_1, N_obj_2)
    :param db:          (N_obj_1/2, )
    :param match_gt:    (N_obj_1, N_obj_2)
    :param missing_gt:  (N_obj_1/2, )
    :return:
    """
    tp = np.sum(aff_mat * match_gt)  # object match correctly detected
    tn = np.sum(
        db * missing_gt
    )  # correctly identified that object doesn't have a match
    fp = np.sum(aff_mat > match_gt)
    fn = np.sum(db > missing_gt)

    return tp, tn, fp, fn


def count_correct_matches_image_pair(
    aff_matrix_list, dustbin_list, obj_match_gt, obj_missing_gt
):
    """

    :param aff_matrix_list:     [2 x (N_1, N_2)]
    :param dustbin_list:        [(N_1,), ..., (N_n,)]
    :param obj_match_gt:        [2 x (N_1, N_2)]
    :param obj_missing_gt:      [(N_1,), ..., (N_n,)]
    :param n_obj:               (2,)
    :return:
    """

    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(len(aff_matrix_list)):  # iterate over e.g. both images
        tp_frame, tn_frame, fp_frame, fn_frame = count_correct_matches(
            aff_matrix_list[i], dustbin_list[i], obj_match_gt[i], obj_missing_gt[i]
        )

        tp += tp_frame
        tn += tn_frame
        fp += fp_frame
        fn += fn_frame

    return tp, tn, fp, fn


def compute_prec_rec_acc(tp, tn, fp, fn, total_gt_matches=None, total_gt_all=None):

    prec = tp / (tp + fp)

    if total_gt_matches is not None:
        rec = (
            tp / total_gt_matches
        )  # make sure that gt matches were counted wrt to both images
    else:
        rec = tp / (tp + fn)

    if total_gt_all is not None:
        acc = (tp + tn) / total_gt_all
    else:
        acc = (tp + tn) / (tp + fp + tn + fn)
    return prec, rec, acc


# --------------------------------------------------
# ---


def eval_quantitative_single_batch_class(data, eval_dict, names, cnfg, save_gt=True):
    """
    Evaluate performance on classification task.

    :param data:        {...}       -> needs to contain name[0], name[1]
    :param eval_dict:   {...}
    :param names:        <string>   -> e.g. ('obj_cls_gt', 'obj_cls_pred')
    :param cnfg:        {'data': {'num_cls_sem_obj': .., ..}, ..}
    :param save_gt:     <bool>      -> allows to save gt classes only once (e.g. for variants of obj. class prediction)
    :return:    >> updated eval_dict
    """

    cls_gt, cls_pred = names
    cls_gt_short = cls_gt.replace("_gt", "")
    cls_pred_short = cls_pred.replace("_pred", "")
    name_short = cls_gt.split("_")[
        0
    ]  # currently only 'obj', but can so easily be extended to eg. relationships

    if "lbl_" + cls_gt not in eval_dict.keys():
        eval_dict[f"total_n_{cls_gt_short}"] = 0.0
        eval_dict[f"lbl_{cls_gt}"] = np.array(())
    if "lbl_" + cls_pred not in eval_dict.keys():
        eval_dict[f"lbl_{cls_pred}"] = np.array(())
        eval_dict[f"score_{cls_pred_short}_acc"] = 0.0

    lbls_cls_gt_onehot = (
        data[cls_gt]
        if len(data[cls_gt].shape) == 2
        else np.squeeze(data[cls_gt], axis=0)
    )
    lbls_cls_pred_prob = (
        data[cls_pred]
        if len(data[cls_pred].shape) == 2
        else np.squeeze(data[cls_pred], axis=0)
    )
    cls_acc, cls_acc_pc, lbls_cls_gt, lbls_cls_pred = class_accuracy(
        lbls_cls_gt_onehot, lbls_cls_pred_prob
    )

    if save_gt:
        eval_dict[f"lbl_{cls_gt}"] = np.append(eval_dict[f"lbl_{cls_gt}"], lbls_cls_gt)
        eval_dict[f"total_n_{cls_gt_short}"] += lbls_cls_gt.shape[0]

    eval_dict[f"lbl_{cls_pred}"] = np.append(
        eval_dict[f"lbl_{cls_pred}"], lbls_cls_pred
    )
    eval_dict[f"score_{cls_pred_short}_acc"] += cls_acc

    return eval_dict


def eval_quantitative_single_batch_pose(data, eval_dict, cnfg, img_dir=""):
    """

    :param data:            {...}   -> needs to contain 'obj_pose_dist/delta_pred'
    :param eval_dict:       {...}
    :param cnfg:            {'data': {'dist_factor': .., ..}, ..}
    :param img_dir:         <string>
    :return:    >> updated eval_dict
    """
    if "score_obj_pos_err" not in eval_dict.keys():
        eval_dict["score_obj_pos_err"] = []
        eval_dict["total_n_obj_pos_filtered"] = 0.0

    dist_factor = cnfg["data"]["dist_factor"]
    img_shape = cnfg["data"]["img_shape_org"]

    # --------------------------------
    # -- 3D Position
    batch_id = 0  # single item in batch anyway (during both validation and test time)
    fid = 0
    if "obj_frame_id" in data.keys():
        fid_obj = (data["obj_frame_id"] == fid).numpy()[batch_id]
    else:
        fid_obj = np.ones((data["obj_ids"].shape[1])).astype(
            np.bool
        )  # consider all objects

    mpa = data["mpa"][batch_id].numpy()
    obj_bb2d = data["obj_bb2d"][batch_id][fid_obj].numpy()

    obj_delta_gt = data["obj_delta2d"][batch_id][fid_obj].numpy()
    obj_dist_gt = data["obj_dist2d"][batch_id][fid_obj].numpy()

    obj_delta_pred = data["obj_pose_delta_pred"][fid_obj].numpy()
    obj_dist_pred = data["obj_pose_dist_pred"][fid_obj].numpy()

    dist_cntr3d, n_valid_pos = position_error(
        [obj_delta_gt, obj_dist_gt],
        [obj_delta_pred, obj_dist_pred],
        obj_bb2d,
        mpa,
        dist_factor,
        img_shape,
    )
    eval_dict["score_obj_pos_err"] += list(dist_cntr3d)
    eval_dict["total_n_obj_pos_filtered"] += n_valid_pos

    return eval_dict


def eval_quantitative_single_batch_rel(data, eval_dict):
    """

    :param data:            {...}   -> needs to contain 'rel_pose_pred'
    :param eval_dict:       {...}
    :return:    >> updated eval_dict
    """
    if "score_rel_err" not in eval_dict.keys():
        eval_dict["score_rel_err"] = []

    # --------------------------------
    # -- Relationship / Distance
    batch_id = 0  # single item in batch anyway (during both validation and test time)
    fid = 0
    if "rel_frame_id" in data.keys():
        fid_rel = (data["rel_frame_id"] == fid)[batch_id]
    else:
        fid_rel = np.ones((data["rel_ids"].shape[1])).astype(
            np.bool
        )  # consider all objects

    rel_pose_gt = data["rel_pose"][batch_id][fid_rel]  # (N_rel)
    rel_pose_pred = data["rel_pose_pred"][
        fid_rel
    ]  # (N_rel, 1) (value in extra dimension)

    diff_rel = list(np.abs(rel_pose_pred[:, 0] - rel_pose_gt))
    eval_dict["score_rel_err"].extend(diff_rel)

    return eval_dict


def eval_quantitative_single_batch_affinity(
    data, eval_dict, img_dir="", full_eval=True
):

    if "n_match_gt" not in eval_dict.keys():
        init_affinity_entries(eval_dict, "basic")

    batch_id = 0  # single item in batch anyway (during both validation and test time)
    fid = 0
    scan_id = data["scan_ids"][0]
    scene_name = eval_dict["general_scene-name_info"][scan_id]

    # ----------------
    # collect gt + pred. data
    obj_ids_match, n_obj, obj_ids_missing, total_gt = structure_gt_data(
        data["obj_ids_match"][batch_id]
    )
    total_gt_matches, total_gt_missing = total_gt
    eval_dict["n_match_gt"] += total_gt_matches
    eval_dict["n_missing_gt"] += total_gt_missing

    affinity_mat_ext = np.exp(data["obj_affinity_mat"])[batch_id]
    affinity_mat = affinity_mat_ext[:-1, :-1]  # throw away dustbin
    dustbin_prob_m = affinity_mat_ext[:-1, -1:]  # (M, 1)
    dustbin_prob_n = affinity_mat_ext[-1:, :-1]  # (1, N)

    # ----------------
    # - prediction
    aff_matrix_list, dustbin_list = max_pred_assignment(affinity_mat_ext)

    tp_max, tn_max, fp_max, fn_max = count_correct_matches_image_pair(
        aff_matrix_list, dustbin_list, 2 * [obj_ids_match], obj_ids_missing
    )
    total_pred_matches = tp_max + fp_max
    res_count_max = [
        ("n-matches", 2 * total_gt_matches),
        ("n-missing", total_gt_missing),
        ("n-tp", tp_max),
        ("n-fp", fp_max),
        ("n-tn", tn_max),
        ("n-fn", fn_max),
    ]
    for k, v in res_count_max:
        eval_dict[f"matching-S_{k}"] += v

    res_match_max = np.maximum(
        aff_matrix_list[0], aff_matrix_list[1]
    )  # for visualization

    # ----------------
    # - visualization
    viz_simple = False
    scan_ids = data["scan_ids"][0].numpy()
    frame_ids = data["frame_ids"].numpy()

    if (
        viz_simple and frame_ids <= 1 and img_dir != ""
    ):  # set arbitrary number to limit number of output images
        img_name = (
            str(data["scan_ids"][fid].numpy())
            + "_frame"
            + str(data["frame_ids"].numpy())
        )
        if True:  # not os.path.exists(img_path):
            scene_rgb = data["scene_rgb"].numpy()
            scene_rgb = np.reshape(
                scene_rgb, (-1, 2, scene_rgb.shape[1], scene_rgb.shape[2], 3)
            )[batch_id]

            fid_obj_1 = np.squeeze(data["obj_frame_id"]) == 0
            obj_bb_1 = np.squeeze(data["obj_bb2d"])[fid_obj_1]
            fid_obj_2 = np.squeeze(data["obj_frame_id"]) == 1
            obj_bb_2 = np.squeeze(data["obj_bb2d"])[fid_obj_2]

            # # Plot final matches (based on max. affinity score)
            match_rgb = viz.draw_sim(
                np.copy(scene_rgb),
                obj_ids_match,
                res_match_max,
                [obj_bb_1, obj_bb_2],
                concat_axis=1,
            )
            img_path = os.path.join(img_dir, "match_max_scene" + img_name + ".png")
            viz.show_image(match_rgb, path=img_path)

    return eval_dict


def eval_quantitative_single_batch_affinity_detections(
    data, input_org, eval_dict, img_dir=""
):

    if "n_match_gt" not in eval_dict.keys():
        init_affinity_entries(eval_dict, "basic")
        for l in [
            "n_obj_gt",
            "n_obj_detect",
            "n_obj_detect_wo_counterpart",
            "n_match_detect",
            "n_missing_detect",
            "matching-S_n-tn_vGT",
            "matching-S_n-fn_vGT",
        ]:
            eval_dict[l] = 0

    def split_obj2frame(obj_data, split_ids):
        return [obj_data[split_ids == i] for i in range(2)]

    def get_gt_match_mtrx(obj_ids):
        n_obj1 = len(obj_ids[0])
        n_obj2 = len(obj_ids[1])
        obj_ids_match = np.zeros((n_obj1, n_obj2))
        for i in range(n_obj1):
            for j in range(n_obj2):
                if obj_ids[0][i] == obj_ids[1][j] and obj_ids[0][i] >= 0:
                    obj_ids_match[i, j] = 1
        return obj_ids_match, n_obj1, n_obj2

    # ----------------
    # collect gt + pred. data
    obj_ids_det = split_obj2frame(data["obj_ids"].numpy(), data["obj_frame_id"].numpy())
    obj_ids_gt = split_obj2frame(
        input_org["obj_ids"].numpy(), input_org["obj_frame_id"].numpy()
    )

    n_obj_detected_wo_counterpart = 0
    for i in range(
        2
    ):  # for each image: count wrongly detected 'objects', i.e. such without counterpart
        n_obj_detected_wo_counterpart += np.sum(
            np.where(
                obj_ids_det[i] < 0,
                np.ones_like(obj_ids_det[i]),
                np.zeros_like(obj_ids_det[i]),
            )
        )
    eval_dict["n_obj_detect_wo_counterpart"] += n_obj_detected_wo_counterpart

    obj_ids_match_gt, _, _ = get_gt_match_mtrx(obj_ids_gt)
    _, n_obj_gt, obj_ids_missing_gt, total_gt = structure_gt_data(obj_ids_match_gt)
    total_matches_gt, total_missing_gt = total_gt
    eval_dict["n_match_gt"] += total_matches_gt
    eval_dict["n_missing_gt"] += total_missing_gt
    eval_dict["n_obj_gt"] += sum(n_obj_gt)

    obj_ids_match_detect, _, _ = get_gt_match_mtrx(
        obj_ids_det
    )  # more stable wrt un-assigned detections than: data['obj_ids_match'][0].numpy()
    ##
    # update: don't count objects without counterpart as 'should get assigned to dustbin'
    n_obj1_detect, n_obj2_detect = obj_ids_match_detect.shape
    n_obj_detect = [n_obj1_detect, n_obj2_detect]

    obj_ids_missing_m_detect = np.ones((n_obj1_detect,)) - np.clip(
        np.sum(obj_ids_match_detect, axis=1), 0, 1
    )  # (M,)  # -> diffuculty about object detections (multiple objects assigned to same gt object
    obj_ids_missing_m_detect = np.where(
        obj_ids_det[0] >= 0,
        obj_ids_missing_m_detect,
        np.zeros_like(obj_ids_missing_m_detect),
    )
    obj_ids_missing_n_detect = np.ones((n_obj2_detect,)) - np.clip(
        np.sum(obj_ids_match_detect, axis=0), 0, 1
    )  # (N,)
    obj_ids_missing_n_detect = np.where(
        obj_ids_det[1] >= 0,
        obj_ids_missing_n_detect,
        np.zeros_like(obj_ids_missing_n_detect),
    )
    obj_ids_missing_detect = [obj_ids_missing_m_detect, obj_ids_missing_n_detect]

    total_matches_detect = np.sum(obj_ids_match_detect)
    total_missing_detect = np.sum(obj_ids_missing_m_detect) + np.sum(
        obj_ids_missing_n_detect
    )
    total_detect = [total_matches_detect, total_missing_detect]
    ##
    # total_matches_detect, total_missing_detect = total_detect
    eval_dict["n_match_detect"] += total_matches_detect
    eval_dict["n_missing_detect"] += total_missing_detect
    eval_dict["n_obj_detect"] += sum(n_obj_detect)

    # -> consider ACTUAL gt dustbin associations
    #    (might be less, e.g. because pairing object was not detected in other image)
    obj_ids_missing_detect_vGT = [
        np.zeros((obj_ids_match_detect.shape[i])) for i in range(2)
    ]
    for frame_id in range(2):
        for oid, obj_id in enumerate(obj_ids_det[frame_id]):
            if obj_id <= -1:
                # update: don't count objects without counterpart as 'should get assigned to dustbin'
                # obj_ids_missing_detect_vGT[frame_id][oid] = 1  # gt for objects w/o counterpart: assigned to dustbin
                continue
            oid_gt = list(obj_ids_gt[frame_id]).index(obj_id)
            obj_ids_missing_detect_vGT[frame_id][oid] = obj_ids_missing_gt[frame_id][
                oid_gt
            ]

        if obj_ids_missing_detect[frame_id].shape[0] > 0:
            assert (
                np.min(
                    obj_ids_missing_detect[frame_id]
                    - obj_ids_missing_detect_vGT[frame_id]
                )
                >= 0
            )
        else:
            assert obj_ids_missing_detect_vGT[frame_id].shape[0] == 0

    # ----------------
    # - prediction
    affinity_mat_ext = np.exp(data["obj_affinity_mat"][0])
    aff_matrix_list, dustbin_list = max_pred_assignment(affinity_mat_ext)

    res_detect = count_correct_matches_image_pair(
        aff_matrix_list,
        dustbin_list,
        2 * [obj_ids_match_detect],
        obj_ids_missing_detect,
    )
    tp_detect, tn_detect, fp_detect, fn_detect = res_detect
    # -> count correct association to dustbin wrt gt (i.e. bad, if an object was detected in only one frame)
    res_vGT = count_correct_matches_image_pair(
        aff_matrix_list,
        dustbin_list,
        2 * [obj_ids_match_detect],
        obj_ids_missing_detect_vGT,
    )
    _, tn_vGT, _, fn_vGT = res_vGT

    res_count_max = [
        ("n-matches", 2 * total_matches_detect),
        ("n-missing", total_missing_detect),
        ("n-tp", tp_detect),
        ("n-fp", fp_detect),
        ("n-tn", tn_detect),
        ("n-fn", fn_detect),
        ("n-tn_vGT", tn_vGT),
        ("n-fn_vGT", fn_vGT),
    ]
    for k, v in res_count_max:
        eval_dict[f"matching-S_{k}"] += v

    res_match_max = np.maximum(
        aff_matrix_list[0], aff_matrix_list[1]
    )  # -> for visualization

    total_pred_matches = tp_detect + fp_detect

    # visualization
    viz_simple = False
    scan_ids = data["scan_ids"][0].numpy()
    frame_ids = data["frame_ids"].numpy()

    if viz_simple and scan_ids <= 10 and frame_ids[0] <= 5 and img_dir != "":
        if True:  # not os.path.exists(aff_img_path):
            scene_rgb = data["scene_rgb"].numpy()

            fid_obj_1 = (data["obj_frame_id"] == 0).numpy()
            obj_bb_1 = data["obj_bb2d"][fid_obj_1]
            fid_obj_2 = (data["obj_frame_id"] == 1).numpy()
            obj_bb_2 = data["obj_bb2d"][fid_obj_2]

            match_rgb = viz.draw_sim(
                np.copy(scene_rgb),
                obj_ids_match_detect,
                res_match_max,
                [obj_bb_1, obj_bb_2],
            )
            match_img_path = os.path.join(
                img_dir, f"match_scene{str(scan_ids)}{str(frame_ids)}.png"
            )
            viz.show_image(match_rgb, path=match_img_path)

    return eval_dict


# --------------------------------------------------
# ---


def compute_match_scores(
    name_in, name_out, eval_dict, eval_dict_new, copy_scores=False, ext=""
):
    if f"{name_in}_n-tp" not in eval_dict.keys():
        return

    if copy_scores:
        for t in ["n-matches", "n-missing", "n-tp", "n-fp", "n-tn", "n-fn"]:
            eval_dict_new[f"{name_in}_{t}"] = eval_dict[f"{name_in}_{t}"]
    prec, rec, acc = compute_prec_rec_acc(
        eval_dict[f"{name_in}_n-tp"],
        eval_dict[f"{name_in}_n-tn"],
        eval_dict[f"{name_in}_n-fp"],
        eval_dict[f"{name_in}_n-fn"],
        total_gt_matches=eval_dict[f"{name_in}_n-matches"],
    )
    f1score = 2 * (rec * prec) / (rec + prec)
    eval_dict_new[f"{name_out}_f1"] = f1score
    eval_dict_new[f"{name_out}_prec"] = prec
    eval_dict_new[f"{name_out}_rec"] = rec


def finalize_eval_scores(eval_dict, print_out=False, log_file=None):
    """

    :param eval_dict:
    :param print_out:
    :param log_file:
    :return:
    """
    eval_dict_new = {}  # necessary as dictionary changes size during iteration

    # -> Summarize object + relationship evaluation
    for k in eval_dict.keys():
        if "score_" in k:
            if "_err" in k:
                eval_dict_new[k + "_mean"] = np.nanmean(eval_dict[k])
                eval_dict_new[k + "_median"] = np.nanmedian(eval_dict[k])
                for th in [0.5, 1.0]:
                    eval_dict_new[f"{k}_acc{th}"] = np.mean(
                        np.asarray(np.asarray(eval_dict[k]) < th, dtype=np.float32)
                    )
            elif "_cls_" in k:
                entity_type = k.split("_")[1]  # -> e.g. 'obj'
                if "_pc" in k:
                    k_tmp = k.replace("score", "listscores")
                    eval_dict_new[k_tmp] = (
                        eval_dict[k] / eval_dict[f"total_n_{entity_type}_cls_pc"]
                    )
                    eval_dict_new[k + "_mean"] = np.nanmean(
                        eval_dict_new[k_tmp]
                    )  # ignores missing classes (in gt)
                else:
                    eval_dict_new[k] = (
                        eval_dict[k] / eval_dict[f"total_n_{entity_type}_cls"]
                    )
            else:
                if "_match_" in k:
                    eval_dict_new[k] = np.nanmean(eval_dict[k])
        if "lbl_" in k or "info_" in k:
            eval_dict_new[k] = eval_dict[k]

    # -> Summarize matching evaluation
    if "matching-S_n-matches" in eval_dict.keys():

        compute_match_scores(
            "matching-S", "score_match", eval_dict, eval_dict_new
        )  # main eval measure

    if print_out:
        print_eval_scores(eval_dict_new, log_file=log_file)

    return eval_dict_new


def finalize_eval_scores_detections(eval_dict):

    eval_dict_new = {}

    for k in eval_dict.keys():
        if k[:2] == "n_":
            eval_dict_new[k] = eval_dict[k]

    # wrt. detections only
    compute_match_scores("matching-S", "score_match_detect", eval_dict, eval_dict_new)

    # wrt. all gt objects
    name_in = "matching-S"
    name_out = "score_match_gt-all"
    total_gt_matches = 2 * eval_dict["n_match_gt"]  # not counted twice yet
    total_gt_all = 2 * eval_dict["n_match_gt"] + eval_dict["n_missing_gt"]
    prec, rec, acc = compute_prec_rec_acc(
        eval_dict[f"{name_in}_n-tp"],
        eval_dict[f"{name_in}_n-tn_vGT"],
        eval_dict[f"{name_in}_n-fp"],
        eval_dict[f"{name_in}_n-fn_vGT"],
        total_gt_matches=total_gt_matches,
        total_gt_all=total_gt_all,
    )
    f1score = 2 * (rec * prec) / (rec + prec)
    eval_dict_new[f"{name_out}_f1"] = f1score
    eval_dict_new[f"{name_out}_prec"] = prec
    eval_dict_new[f"{name_out}_rec"] = rec

    return eval_dict_new


def print_eval_scores(eval_dict, log_file=None):
    """

    :param eval_dict:   {...}
    :param log_file:    <string>
    :return:
    """
    for k in eval_dict.keys():
        if "score_" in k or "total_n_obj" in k or "matching_res" in k:
            if type(eval_dict[k]) == np.ndarray:
                continue

            # log file or simple output
            if log_file is not None:
                if "matching_res" in k:
                    log_file.write(" -- {}:".format(k) + str(eval_dict[k]))
                else:
                    log_file.write("{:16s}:\t{:.3f}".format(k, eval_dict[k]))
            else:
                if "matching_res" in k:
                    print(" -- {}:".format(k) + str(eval_dict[k]))
                else:
                    print(" -- {:16s}:\t{:.3f}".format(k, eval_dict[k]))
