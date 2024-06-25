"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

import os

from utils.tf_funcs import *


# ---- Handle individual summary entries


def loss_to_summary(name, value):
    """
    summary writer:     <tf object
    name:               <string>
    value:              <float>
    """
    tf.summary.scalar(name, value)


def stats_to_summary(name, value):
    tf.summary.scalar(name, tf.reduce_mean(value))
    tf.summary.scalar(name + "_std", tf.math.reduce_std(value))


def img_to_summary(name, imgs):
    if imgs.dtype != tf.uint8:
        imgs = tf.clip_by_value(imgs, 0.0, 1.0)
        imgs = tf.cast(255 * imgs, tf.uint8)
    imgs = tf.reshape(imgs, (-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1]))
    tf.summary.image(name, imgs, max_outputs=1)


def depth_to_summary(name, depth):
    if tf.shape(depth)[-1] != 1:
        depth = tf.expand_dims(depth, -1)
    depth_min = tf.math.reduce_min(input_tensor=depth)
    depth_max = tf.math.reduce_max(input_tensor=depth)
    depth = (depth - depth_min) / (depth_max - depth_min)

    img_to_summary(name, depth)


# ---- Manage entire model output


def summarize_all_losses(loss_dict, mode, log_file=None):
    for k, v in loss_dict.items():
        loss_flag = "loss" in k or ("l_" == k[:2])
        if loss_flag:
            loss_to_summary("loss/" + mode + "/" + k, v)
        elif "ctrl" in k:
            loss_to_summary("loss-ctrl/" + k.replace("-ctrl", ""), v)

        if log_file is not None and loss_flag:
            log_file.write(k + ": \t" + str(v.numpy()))
        elif loss_flag:
            print(k + ": " + str(v.numpy()))


def summarize_all_simple(dict, name):
    for k, v in dict.items():
        if k[:2] == "w-":
            tf.summary.scalar(name + "/" + k, tf.reduce_mean(v))
        elif k[:6] == "score_" and k[-3:] != "_pc":
            tf.summary.scalar(name + "/" + k[6:], tf.reduce_mean(v))
        elif k[:4] == "var_":
            tf.summary.scalar(name + "/" + k, tf.reduce_mean(v))


def summarize_all_images(data_dict, types=[]):

    for t in types:
        name = t

        if t not in data_dict.keys():
            continue

        if "rgb" in t:
            if name == "rgb_in":
                name = "0_input"
            img_to_summary(name, data_dict[t])

        elif "msk" in t:
            depth_to_summary(name, data_dict[t])

        elif "depth" in t:
            depth_to_summary(name, data_dict[t])

        elif "sem" in t:
            img_to_summary(name, data_dict[t])

        elif "inst" in t:
            img_to_summary(name, data_dict[t])


def summarize_all(
    summary_writer, epoch, summary_data, imgs_types=[], log_file=None, mode="train"
):

    with summary_writer.as_default(step=epoch):
        if "losses" in summary_data.keys():
            summarize_all_losses(summary_data["losses"], mode, log_file)
        if "inputs" in summary_data.keys():
            summarize_all_images(summary_data["inputs"], imgs_types)
        if "outputs" in summary_data.keys():
            summarize_all_images(summary_data["outputs"], imgs_types)
        if "eval" in summary_data.keys():
            summarize_all_simple(summary_data["eval"], "val")
        if "params" in summary_data.keys():
            summarize_all_simple(summary_data["params"], "params")
