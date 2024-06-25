"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

from numpy import pi

# -------------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------------

training = {
    "max_epoch": 350,
    "save_epoch": 5,
    "batch_size": 32,
    "learning_rate": 0.0001,
}

# -------------------------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------------------------

model = {
    "name": "romnet-matchVflex",
    "backbone": "ResNet34",  # ['ResNet34', 'use_pretrained']  -> ResNet34 for pre-computing
    "featnet_pose_enc_layers": [32, 64, 128],
    "featnet_prop_enc_layers": [
        512,
        256,
        128,
        128,
    ],
    "featnet_sim_layers": [256, 256],
    "agnn_layers": ["self", "cross"] * 2,
    "sinkhorn_iters": 10,
    "l-weights": {
        "obj-cls": 1.0,
        "obj-pos": 0.1,
        "rel-dist": 0.1,
        "contrastive": 0.0,
        "affinity": 1.0,
    },
}


# -------------------------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------------------------

data = {
    "name": "hypersim",
    "img_shape_org": (768, 1024),
    "img_shape": (384, 512),
    "img_shape_coarse": (384, 512),
    "img_shape_obj": (128, 128),
    "lvl_difficulty": {
        "train": [0, 1, 2],
        "val": [0],
        "test": [0, 1, 2],
    },
    "cls_labeling": "nyu40",
    "num_cls_sem_obj": 40,
    "bins_dist": [[-0.5, 0.5]],
    "dist_factor": 10.0,
    "bins_orient": [
        [(i - 6 / 2) * float(2 * pi / 6), (i - 6 / 2 + 1) * float(2 * pi / 6)]
        for i in range(6)
    ],  # orientation bin ranges from -np.pi to np.pi, 60 degrees width for each bin.
    "n_smpl_obj": 5,
    "max_n_smpl_obj": 40,
    "n_smpl_keypoint": 100,
    "augmentation": ["gauss"],
    "use_pretrained": False,  # -> False for pre-computing
}


# -------------------------------------------------------------------------------------------
# Dictionary Output
# -------------------------------------------------------------------------------------------

cnfg_dict = {"training": training, "model": model, "data": data}
