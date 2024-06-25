"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import cv2
from PIL import ImageEnhance


class DataAugmetation:
    def __init__(self, types):
        self.types = types

    # -- variants of augmentation
    def bb_noise(self, obj_2dbbs, img_size, frac_bb=1 / 3.0):
        """

        :param obj_2dbbs:
        :param img_size:    (height <int>, width <int>)
        :param frac_bb:     <float>
        :return:
        """
        img_height, img_width = img_size

        bb_length = np.asarray(
            [[o[2] - o[0], o[3] - o[1]] for o in obj_2dbbs], dtype=np.float32
        )

        eps = (np.tile(bb_length, (1, 2)) * frac_bb).astype(np.int32)
        noise = np.random.randint(eps) - (eps / 2).astype(np.int32)

        obj_2dbbs_w_noise = obj_2dbbs + noise
        obj_2dbbs = np.stack(
            [
                np.clip(obj_2dbbs_w_noise[:, 0], 0, img_height),
                np.clip(obj_2dbbs_w_noise[:, 1], 0, img_width),
                np.clip(obj_2dbbs_w_noise[:, 2], 0, img_height),
                np.clip(obj_2dbbs_w_noise[:, 3], 0, img_width),
            ],
            axis=-1,
        )

        return obj_2dbbs

    def enhance_rgb(self, rgb_img):
        """

        :param rgb_img:
        :return:
        """
        eps_brightness = np.clip(np.random.normal(1.0, 0.25), 0.25, 1.75)
        enhancer = ImageEnhance.Brightness(rgb_img)
        rgb_img = enhancer.enhance(eps_brightness)

        eps_contrast = np.clip(np.random.normal(1.0, 0.25), 0.25, 2.0)
        enhancer = ImageEnhance.Contrast(rgb_img)
        rgb_img = enhancer.enhance(eps_contrast)

        eps_sharpness = np.clip(np.random.normal(1.0, 0.25), 0.0, 2.0)
        enhancer = ImageEnhance.Sharpness(rgb_img)
        rgb_img = enhancer.enhance(eps_sharpness)

        eps_color = np.clip(np.random.normal(1.0, 0.25), 0.0, 2.0)
        enhancer = ImageEnhance.Color(rgb_img)
        rgb_img = enhancer.enhance(eps_color)

        eps_hue = np.clip(np.abs(np.random.normal(0.0, 0.125)), 0.0, 0.5)
        eps_hue_col = 255.0 * np.random.uniform(0.0, 1.0, (1, 1, 3))
        rgb_img = (1.0 - eps_hue) * np.asarray(
            rgb_img, dtype=np.float32
        ) + eps_hue * eps_hue_col
        rgb_img = np.clip(rgb_img, 0.0, 255.0)

        return rgb_img

    def gauss_noise(self, viz_data, eps_type=0):
        """

        :param viz_data:    ([N_obj,] F) / ([N_obj,] H_obj, W_obj, 3)
        :param eps_type:    <int>
        :return:
        """

        if eps_type == 0:  # constant gaussian noise
            eps = 0.01
        else:
            eps = np.clip(np.abs(np.random.normal(0.0, 0.05)), 0.0, 0.1)
        viz_data += np.random.normal(0.0, eps, viz_data.shape)

        return viz_data

    # --
