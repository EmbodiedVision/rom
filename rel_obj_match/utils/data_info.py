"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

# --------------------------------------------------
# ---  Label Names

NYU40_Label_Names = [
    "wall",  # 1
    "floor",  # 2
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",  # 8
    "window",  # 9
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",  # 15
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floor mat",  # 20
    "clothes",
    "ceiling",  # 22
    "books",
    "refrigerator",
    "television",
    "paper",
    "towel",
    "shower curtain",
    "box",
    "whiteboard",  # 30
    "person",
    "night stand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
    "otherstructure",  # 38
    "otherfurniture",  # 39
    "otherprop",  # 40
]

NYU40_Structure_Categories = [
    # wall, floor, window, door, otherstructure, otherprob
    0,
    1,
    7,
    8,
    37,
    39,
]


# --------------------------------------------------
# ---


def get_subset_categories(dataset, subset):
    if (dataset == "nyu40" or dataset == "NYU40") and subset == "structure":
        return NYU40_Structure_Categories
    else:
        print(
            [
                "ERROR: data_info.get_subset_categories(): Dataset {} is unknown.".format(
                    dataset
                )
            ]
        )
        exit(1)


def get_class_names(dataset):
    if dataset == "nyu40" or dataset == "NYU40":
        return NYU40_Label_Names
    else:
        print(
            [
                "ERROR: data_info.get_class_names(): Dataset {} is unknown.".format(
                    dataset
                )
            ]
        )
        exit(1)
