"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.data_info import *


# --------------------------------------------------
# --- Visualize 2D images


def np_to_pil(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
    img = (255 * img).astype(np.uint8)
    img = Image.fromarray(img)
    return img


def show_image(img, path=None):
    """
    :param img:
    :param path:
    :return:
    """

    img = np_to_pil(img)

    if path is None:
        img.show()
    else:
        try:
            img.save(path)
            print("Saved image to {}.".format(path))
        except IOError:
            print("Path {} does not exist.".format(path))


def prepare_image(img, type):
    if type == "rgb":
        if np.max(img) > 1.5:  # ignore color noise
            img = (1.0 / 255) * img
        img = np.clip(img, 0.0, 1.0)
    elif type == "depth":
        img = np.nan_to_num(img, nan=np.nanmax(img))
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        img = np.tile(img, (1, 1, 3))
    else:
        print("[ERROR] viz.prepare_image(), unknown image type {}".format(type))

    return img


def show_img_grid(imgs, types, path=None):
    """
    :param imgs: [[np-images]] (so far, images need to have same dimension)
    :param types: [[img-types (str)]] -> same dimension as imgs!
    :return:
    """
    final = []
    for i in range(len(imgs)):
        final.append([])
        for j in range(len(imgs[0])):
            final[i].append([])
            final[i][j] = prepare_image(img=imgs[i][j], type=types[i][j])
    final = np.asarray(final)
    final = np.transpose(final, (0, 2, 1, 3, 4))
    final = np.reshape(
        final, (final.shape[0] * final.shape[1], final.shape[2] * final.shape[3], 3)
    )

    show_image(final, path=path)


def draw_bb(img, bounding_boxes, bb_norm=True, c=2, sem_lbls=None):
    h, w, _ = img.shape

    if bb_norm:
        bounding_boxes = np.floor(
            np.stack(
                [
                    bounding_boxes[:, 0] * h,
                    bounding_boxes[:, 1] * w,
                    bounding_boxes[:, 2] * h,
                    bounding_boxes[:, 3] * w,
                ],
                axis=-1,
            )
        ).astype(np.int)

    for i, bb in enumerate(bounding_boxes):

        if sem_lbls is None or len(sem_lbls) == 0:
            sem_col = np.array([0.0, 0.0, 0.0])
        else:
            sem_col = np.array(create_color_palette("nyu40")[sem_lbls[i] + 1]) / 255.0

        img[bb[0] : bb[2], max(bb[1] - (c - 1), 0) : max(bb[1] + c, 0)] = sem_col
        img[bb[0] : bb[2], min([bb[3] - (c - 1), w - 1]) : min([bb[3] + c, w])] = (
            sem_col
        )
        img[max(bb[0] - (c - 1), 0) : max(bb[0] + c, 0), bb[1] : bb[3]] = sem_col
        img[min(bb[2] - (c - 1), h) : min(bb[2] + c, h), bb[1] : bb[3]] = sem_col

        if c > 0:
            bb_cntr = [
                int(bb[0] + (bb[2] - bb[0]) / 2.0),
                int(bb[1] + (bb[3] - bb[1]) / 2),
            ]
            img[
                np.max([bb_cntr[0] - c, 0]) : np.min([bb_cntr[0] + c, h - 1]),
                np.max([bb_cntr[1] - c, 0]) : np.min([bb_cntr[1] + c, w - 1]),
            ] = np.array([1.0, 1.0, 1.0])

    return img


def draw_cntrs(img, cntr_proj, bb, bb_norm=True, dist=None, sem_lbls=None, size=15):
    h, w, _ = img.shape

    cntr_proj = cntr_proj.astype(int)
    if bb_norm:
        bb = np.floor(
            np.stack([bb[:, 0] * h, bb[:, 1] * w, bb[:, 2] * h, bb[:, 3] * w], axis=-1)
        ).astype(np.int)

    if dist is None:
        dist = [5 for _ in cntr_proj]

    for i, c in enumerate(cntr_proj):
        if sem_lbls is None or len(sem_lbls) == 0:
            sem_col = np.array([0.0, 0.0, 0.0])
        else:
            sem_col = np.array(create_color_palette("nyu40")[sem_lbls[i] + 1]) / 255.0

        if (0 <= c[0] < h) and (0 <= c[1] < w):
            s = max(int(size / dist[i]), 2)
            img[
                np.max([c[0] - s, 0]) : np.min([c[0] + s, h - 1]),
                np.max([c[1] - s, 0]) : np.min([c[1] + s, w - 1]),
            ] = sem_col
        elif bb is not None:
            s = max(int(size / dist[i]), 2)  # int(max(size, 5))
            if c[0] < 0:
                img[0:s, bb[i, 1] : bb[i, 3] - 1] = sem_col
            if c[0] >= h:
                img[h - s : h, bb[i, 1] : bb[i, 3] - 1] = sem_col
            if c[1] < 0:
                img[bb[i, 0] : bb[i, 2], 0:s] = sem_col
            if c[1] >= w:
                img[bb[i, 0] : bb[i, 2], w - s : w] = sem_col

    return img


def draw_keypoint_matches(imgs, matches, obj_bb=None, filter=False):
    h, w, _ = imgs[0].shape
    b = 5  # border between two images
    img_size_ratio = np.asarray([[[h / (768.0), w / 1024.0]]])

    matches = (img_size_ratio * matches).astype(np.int)

    if filter:
        matches_filtered = []
        for m_id in range(matches.shape[1]):
            in_bb_1, in_bb_2 = False, False
            m0 = matches[0, m_id]
            m1 = matches[1, m_id]

            for bb in obj_bb[0]:
                if bb[0] * h <= m0[0] <= bb[2] * h and bb[1] * w <= m0[1] <= bb[3] * w:
                    in_bb_1 = True
                    break
            for bb in obj_bb[1]:
                if bb[0] * h <= m1[0] <= bb[2] * h and bb[1] * w <= m1[1] <= bb[3] * w:
                    in_bb_2 = True
                    break
            if in_bb_1 and in_bb_2:
                matches_filtered.append(matches[:, m_id])
        if len(matches_filtered) > 0:
            matches = np.stack(matches_filtered, axis=1)
        else:
            matches = np.zeros((2, 0, 2))

    if obj_bb is not None:
        imgs = [draw_bb(imgs[i], obj_bb[i]) for i in range(2)]
    img_cmb = concat_imgs(imgs[0], imgs[1], b=b)

    col = np.array([0.0, 0.0, 0.0])
    s = 3

    for m_id in range(matches.shape[1]):
        m0 = matches[0, m_id]
        m1 = matches[1, m_id] + [0, w + b]

        img_cmb[m0[0] - s : m0[0] + s, m0[1] - s : m0[1] + s] = col
        img_cmb[m1[0] - s : m1[0] + s, m1[1] - s : m1[1] + s] = col

    img_cmb = np_to_pil(img_cmb)
    draw = ImageDraw.Draw(img_cmb)

    s = 1

    for m_id in range(matches.shape[1]):
        m0 = matches[0, m_id]
        m1 = matches[1, m_id] + [0, w + b]

        draw.line([m0[1], m0[0], m1[1], m1[0]], fill="black", width=s)

    img_res = np.asarray(img_cmb, dtype=np.float32) / 255.0
    return img_res


def draw_bb_with_cntrs(
    img, cntr_proj, bb, bb_norm=True, dist=None, sem_lbls=None, size=10
):
    img = draw_cntrs(
        img, cntr_proj, bb, bb_norm=bb_norm, dist=dist, sem_lbls=sem_lbls, size=size
    )
    img = draw_bb(img, bb, bb_norm=bb_norm, sem_lbls=sem_lbls, c=int(size / 3))
    return img


def draw_sim(
    img, obj_ids_match, obj_sim_mat, obj_bb, bb_norm=True, th=0.1, concat_axis=1, gap=5
):
    _, h, w, _ = img.shape
    ll_width = int(h / 100)  # 3
    img = [draw_bb(img[i], obj_bb[i]) for i in range(2)]
    img_cmb = concat_imgs(img[0], img[1], b=gap, axis=concat_axis)
    heigh_off = img_cmb.shape[0] - h
    width_off = img_cmb.shape[1] - w
    img_cmb = np_to_pil(img_cmb)
    draw = ImageDraw.Draw(img_cmb)
    n_obj_1, n_obj_2 = obj_ids_match.shape
    if bb_norm:
        obj_bb = [
            np.floor(
                np.stack(
                    [
                        obj_bb[i][:, 0] * h,
                        obj_bb[i][:, 1] * w,
                        obj_bb[i][:, 2] * h,
                        obj_bb[i][:, 3] * w,
                    ],
                    axis=-1,
                )
            ).astype(np.int)
            for i in range(2)
        ]

    for i in range(n_obj_1):
        i_bb = obj_bb[0][i]
        i_cntr = (
            int(i_bb[1] + 0.5 * (i_bb[3] - i_bb[1])),
            int(i_bb[0] + 0.5 * (i_bb[2] - i_bb[0])),
        )
        for j in range(n_obj_2):
            if concat_axis == 0:
                j_bb = obj_bb[1][j] + np.asarray([heigh_off, 0.0, heigh_off, 0.0])
                j_cntr = (
                    int(j_bb[1] + 0.5 * (j_bb[3] - j_bb[1])),
                    int(j_bb[0] + 0.5 * (j_bb[2] - j_bb[0])),
                )
            else:  # concat_axis == 1:
                j_bb = obj_bb[1][j] + np.asarray([0.0, width_off, 0.0, width_off])
                j_cntr = (
                    int(j_bb[1] + 0.5 * (j_bb[3] - j_bb[1])),
                    int(j_bb[0] + 0.5 * (j_bb[2] - j_bb[0])),
                )
            sim_ij = obj_sim_mat[i, j]
            if sim_ij > th:
                if obj_ids_match[i, j]:
                    draw.line(
                        [i_cntr, j_cntr],
                        fill=(int(255 * (1 - sim_ij)), 255, int(255 * (1 - sim_ij))),
                        width=ll_width,
                    )  # tp
                else:
                    draw.line(
                        [i_cntr, j_cntr],
                        fill=(255, int(255 * (1 - sim_ij)), int(255 * (1 - sim_ij))),
                        width=ll_width,
                    )  # tp
            else:
                if obj_ids_match[i, j]:
                    draw.line([i_cntr, j_cntr], fill="yellow", width=ll_width)  # fn

    img_res = np.asarray(img_cmb, dtype=np.float32) / 255.0
    return img_res


def concat_imgs(img1, img2, b=5, axis=1):
    if axis == 0:
        img = np.concatenate([img1, np.ones((b, img2.shape[1], 3)), img2], axis=axis)
    if axis == 1:
        img = np.concatenate([img1, np.ones((img2.shape[0], b, 3)), img2], axis=axis)
    return img


# --------------------------------------------------
# --- Evaluation Visualization


def plot_to_np(fig):
    """
    -> https://martin-mundt.com/tensorboard-figures/

    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X).
    img = img / 255.0

    return img


def plot_xy_curve(x_data, y_data, x_label, y_label, name, pnt_data=None, path=None):
    """

    :param x_data:      [N]+
    :param y_data:      [N]
    :param x_label:     <string>
    :param y_label:     <string>
    :param name:        <string>
    :param pnt_data:    [2,]
    :param path:        <string>
    :return:
    """
    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if pnt_data is not None:
        plt.plot(pnt_data[0], pnt_data[1], marker="o", color="red")

    if path is not None:
        plt.savefig(os.path.join(path, name))
    else:
        plt.show()

    plt.clf()


def plot_bars(data, labels_group, title, name, path=None):

    n = len(labels_group)
    labels_cat = list(data[0].keys())
    data_update = [[data[i][l] for l in labels_cat] for i in range(len(data))]

    x = np.arange(len(labels_cat))  # the label locations
    width = 0.8 / n  # the width of the bars

    fig, ax = plt.subplots(figsize=(15, 10))
    for i in range(n):
        rects = ax.bar(
            x - (n * width) / 2.0 + (i + 0.5) * width,
            data_update[i],
            width,
            label=labels_group[i],
        )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels_cat)))
    ax.set_xticklabels([f"{l}" for l in labels_cat])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.legend()

    fig.tight_layout()

    if path is not None:
        plt.savefig(os.path.join(path, name))
    else:
        plt.show()
    plt.clf()


def plot_histogramm(data, lables, title, bins=None, path=None):

    plt.hist(data, bins=bins, label=lables)
    plt.legend(loc="upper right")
    plt.title(title)

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()


# --------------------------------------------------
# ---  Color Palettes


def mapping_with_list(img, out_shape, transfer_list):
    res = np.zeros(out_shape)
    for idx, value in enumerate(transfer_list):
        res[img == idx] = value

    return res


# color palette for nyu40 labels
def _create_color_palette_nyu40():
    return [
        (0, 0, 0),
        (174, 199, 232),  # wall
        (152, 223, 138),  # floor
        (31, 119, 180),  # cabinet
        (255, 187, 120),  # bed
        (188, 189, 34),  # chair
        (140, 86, 75),  # sofa
        (255, 152, 150),  # table
        (214, 39, 40),  # door
        (197, 176, 213),  # window
        (148, 103, 189),  # bookshelf
        (196, 156, 148),  # picture
        (23, 190, 207),  # counter
        (178, 76, 76),
        (247, 182, 210),  # desk
        (66, 188, 102),
        (219, 219, 141),  # curtain
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14),  # refrigerator
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),  # shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  # toilet
        (112, 128, 144),  # sink
        (96, 207, 209),
        (227, 119, 194),  # bathtub
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),  # otherfurn
        (100, 85, 144),
    ]


def _create_color_palette_inst():
    tmp = list(cm.get_cmap("Set2").colors)
    return tmp * 10


def create_color_palette(dataset):
    if dataset == "nyu40":
        return _create_color_palette_nyu40()
    elif dataset == "instance":
        return _create_color_palette_inst()
    else:
        print(
            [
                "ERROR: viz.create_color_palette(): Dataset {} is unknown.".format(
                    dataset
                )
            ]
        )
        exit(1)
