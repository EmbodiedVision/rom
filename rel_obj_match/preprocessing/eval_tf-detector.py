"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Cathrin Elich, cathrin.elich@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

The source code in this file is part of ROM and licensed under the MIT license 
found in the LICENSE.md file in the root directory of this source tree.
"""

import os
import argparse
from datetime import datetime
import time
import pickle
import random
import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import sys

sys.path.append("/is/sg2/celich/projects/other/Tensorflow/models/research")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


# GPU should not be allocated entirely at beginning
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if len(gpu_devices) > 0:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)


CLASS_NAMES_COCO = [
    "BG",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

CLASS_NAMES_NYU40 = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floor mat",
    "clothes",
    "ceiling",
    "books",
    "refrigerator",
    "television",
    "paper",
    "towel",
    "shower curtain",
    "box",
    "whiteboard",
    "person",
    "night stand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
    "otherstructure",
    "otherfurniture",
    "otherprop",
]

PATH_TO_KERAS_DATA = "/is/sg2/celich/.keras/datasets"  # some models are here as well

MIN_SCORE_THRESHOLD = 0.3
MAX_IOU_THRESHOLD = 0.9

IMAGE_HEIGHT, IMAGE_WIDTH = 768, 1024

NYU40_CATEGORY_INDEX = None
COCO_CATEGORY_INDEX = None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="hypersim", help="Data Set [default: hypersim, (...)]"
    )
    parser.add_argument(
        "--data_dir",
        default="/is/rg/ev/scratch/celich/data/Hypersim",
        help="Data dir [default: /is/rg/ev/scratch/celich/data/Hypersim]",
    )
    parser.add_argument(
        "--model", default="EfficientDet-D7", help="[default: EfficientDet-D7]"
    )
    parser.add_argument(
        "--run_on_cluster",
        type=int,
        choices=(0, 1),
        default=0,
        help="indicator whether job is run on cluster [0 or 1; default=0]",
    )
    parser.add_argument(
        "--subset_only",
        type=int,
        choices=(0, 1),
        default=0,
        help="indicator whether only subset shoul be processed (e.g. for debugging) [0 or 1; default=0]",
    )
    parser.add_argument(
        "--check_missing_only",
        type=int,
        choices=(0, 1),
        default=0,
        help="indicator whether job is run on cluster [0 or 1; default=0]",
    )
    return parser.parse_args()


def map_coco2nyu_lbls(class_ids):
    mapped_lbls = []
    mapped_ids = []
    for c in class_ids:
        class_name = COCO_CATEGORY_INDEX[c]["name"]
        if class_name == "dining table":
            class_name = "table"
        elif class_name == "couch" or class_name == "bench":
            class_name = "sofa"
        elif (
            class_name == "backpack"
            or class_name == "handbag"
            or class_name == "suitcase"
        ):
            class_name = "bag"
        elif class_name == "tv":
            class_name = "television"
        elif class_name == "book":
            class_name = "books"

        if class_name in CLASS_NAMES_NYU40:
            mapped_lbls.append(class_name)
            mapped_ids.append(CLASS_NAMES_NYU40.index(class_name))
        else:
            mapped_lbls.append("otherprop")
            mapped_ids.append(39)  # other prob

    return mapped_lbls, np.asarray(mapped_ids)


def get_iou(bb1, bb2):
    union_bb = [
        min(bb1[0], bb2[0]),
        min(bb1[1], bb2[1]),
        max(bb1[2], bb2[2]),
        max(bb1[3], bb2[3]),
    ]
    union = (union_bb[2] - union_bb[0]) * (union_bb[3] - union_bb[1])
    intersection_bb = [
        max(bb1[0], bb2[0]),
        max(bb1[1], bb2[1]),
        min(bb1[2], bb2[2]),
        min(bb1[3], bb2[3]),
    ]
    intersection = (max(0, intersection_bb[2] - intersection_bb[0])) * max(
        0, (intersection_bb[3] - intersection_bb[1])
    )
    iou = intersection / union
    return iou


# --------------------------------------------------
# --- Main Evaluation


def get_detections(detect_fn, output_dir):
    time_start_str = "{:%d.%m_%H:%M}".format(datetime.now())
    print(f"**** start: {time_start_str} ****")

    raw_data_dir = os.path.join(FLAGS.data_dir, "Hypersim")
    scene_name_list = os.listdir(raw_data_dir)
    if SUBSET_ONLY:
        scene_name_list = scene_name_list[:1]
    if RUN_ON_CLUSTER:
        random.shuffle(
            scene_name_list
        )  # for faster evaluation on cluster when starting several jobs

    if NYU40_CATEGORY_INDEX is not None:
        category_index = NYU40_CATEGORY_INDEX
    else:
        category_index = COCO_CATEGORY_INDEX

    output_dir_detections = os.path.join(output_dir, "detections")
    output_dir_imgs = os.path.join(output_dir, "imgs")
    if not os.path.exists(output_dir_detections):
        os.mkdir(output_dir_detections)
        os.mkdir(output_dir_imgs)

    n_processed = 0
    n_img_all = 0

    for scene_name in scene_name_list:
        print(scene_name)

        cur_output_folder_scene_detections = os.path.join(
            output_dir_detections, scene_name
        )
        cur_output_folder_scene_imgs = os.path.join(output_dir_imgs, scene_name)
        if not os.path.exists(cur_output_folder_scene_detections):
            os.mkdir(cur_output_folder_scene_detections)
            os.mkdir(cur_output_folder_scene_imgs)

        cam_dirs = [
            d
            for d in os.listdir(os.path.join(raw_data_dir, scene_name, "images"))
            if "final_preview" in d
        ]

        if RUN_ON_CLUSTER:
            random.shuffle(cam_dirs)

        for cam_dir in cam_dirs:
            cam_name = "_".join(cam_dir.split("_")[1:3])

            cur_output_folder_cam_detections = os.path.join(
                cur_output_folder_scene_detections, cam_name
            )
            cur_output_folder_cam_imgs = os.path.join(
                cur_output_folder_scene_imgs, cam_name
            )
            if not os.path.exists(cur_output_folder_cam_detections):
                os.mkdir(cur_output_folder_cam_detections)
                os.mkdir(cur_output_folder_cam_imgs)

            img_name_list = [
                i
                for i in os.listdir(
                    os.path.join(raw_data_dir, scene_name, "images", cam_dir)
                )
                if ".tonemap.jpg" in i
            ]

            if RUN_ON_CLUSTER:
                random.shuffle(img_name_list)

            for img_name in img_name_list:

                n_img_all += 1

                out_path_detections = os.path.join(
                    cur_output_folder_cam_detections,
                    img_name.replace(".tonemap.jpg", ".pkl"),
                )
                out_path_img = os.path.join(
                    cur_output_folder_cam_imgs, img_name.replace(".tonemap", "")
                )
                if os.path.exists(out_path_detections) and not SUBSET_ONLY:
                    continue

                n_processed += 1
                if CHECK_MISSING_ONLY:
                    print(scene_name, cam_name, img_name, out_path_img)
                    continue

                # Load image
                img_path = os.path.join(
                    raw_data_dir, scene_name, "images", cam_dir, img_name
                )
                image_np = np.array(Image.open(img_path))

                # Run detection
                input_tensor = tf.convert_to_tensor(image_np)
                input_tensor = input_tensor[tf.newaxis, ...]

                detections = detect_fn(input_tensor)

                num_detections = int(detections.pop("num_detections"))
                interesting_keys = [
                    "detection_boxes",
                    "detection_scores",
                    "detection_classes",
                ]
                detections = {
                    key: value[0, :num_detections].numpy()
                    for key, value in detections.items()
                    if key in interesting_keys
                }
                detections["num_detections"] = num_detections
                detections["detection_classes"] = detections[
                    "detection_classes"
                ].astype(np.int64)

                # filter detections if to high overlap exists
                num_detections_filtered = 0
                detection_boxes = []
                detection_scores = []
                detection_classes = []
                for i in range(num_detections):
                    valid = True
                    for j in range(num_detections):
                        if i == j:
                            continue
                        iou = get_iou(
                            detections["detection_boxes"][i],
                            detections["detection_boxes"][j],
                        )
                        if iou > MAX_IOU_THRESHOLD and (
                            detections["detection_scores"][i]
                            < detections["detection_scores"][i]
                            or (
                                i > j
                                and detections["detection_scores"][i]
                                == detections["detection_scores"][i]
                            )
                        ):
                            valid = False
                    if valid:
                        detection_boxes.append(detections["detection_boxes"][i])
                        detection_scores.append(detections["detection_scores"][i])
                        detection_classes.append(detections["detection_classes"][i])
                        num_detections_filtered += 1
                detections["detection_boxes"] = np.asarray(detection_boxes)
                detections["detection_scores"] = np.asarray(detection_scores)
                detections["detection_classes"] = np.asarray(detection_classes)
                detections["num_detections"] = num_detections_filtered

                # detections: {'num_detections': (N,),
                #               'detection_boxes': (BS, N, 4),
                #               'detection_scores': (BS, N),
                #               'detection_classes': (BS, N)}
                with open(out_path_detections, "wb") as f:
                    pickle.dump(detections, f)

                # Visualize results
                if "0000" in img_name:
                    image_np_with_detections = image_np.copy()
                    viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections["detection_boxes"],
                        detections["detection_classes"],
                        detections["detection_scores"],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=200,
                        min_score_thresh=MIN_SCORE_THRESHOLD,
                        agnostic_mode=False,
                    )
                    Image.fromarray(image_np_with_detections).save(out_path_img)

    if CHECK_MISSING_ONLY:
        print(
            f"[DONE] get_detections(): total #(still missing images) = {n_processed}/{n_img_all}"
        )
    else:
        print(
            f"[DONE] get_detections(): total #(processed images) = {n_processed}/{n_img_all}"
        )


def eval_detections(data_dir, output_dir):
    time_start_str = "{:%d.%m_%H:%M}".format(datetime.now())
    print(f"**** start: {time_start_str} ****")

    # Assignments between detections and gt objects
    assignments = {}

    if NYU40_CATEGORY_INDEX is None:
        classes_names_coco2nyu40, classes_ids_coco2nyu40 = map_coco2nyu_lbls(
            list(COCO_CATEGORY_INDEX.keys())
        )
        classes_names_intersect = list(set(classes_names_coco2nyu40))
        classes_ids_intersect = [
            CLASS_NAMES_NYU40.index(n) for n in classes_names_intersect
        ]
        classes_ids_intersect.sort()
        classes_names_intersect = [CLASS_NAMES_NYU40[i] for i in classes_ids_intersect]
        assignments["classes_intersect"] = {
            "names": classes_names_intersect,
            "ids": classes_ids_intersect,
        }
        cur_cat_index = COCO_CATEGORY_INDEX
    else:
        classes_ids_intersect = list(range(40))
        classes_names_intersect = CLASS_NAMES_NYU40
        cur_cat_index = NYU40_CATEGORY_INDEX

    # criteria are all wrt. gt
    detected_classes = np.zeros(len(classes_names_intersect))
    missing_classes = np.zeros(len(classes_names_intersect))

    detected_sizes = []
    missing_sizes = []

    detect_total_number_per_image = []
    missing_total_number_per_image = []
    wrong_total_number_per_image = []
    total_number_per_image = []
    detect_total_number_per_image_per_object = []
    missing_total_number_per_image_per_object = []

    if SUBSET_ONLY:
        split_sets = ["train"]
    else:
        split_sets = ["val", "test", "train"]
    for split in split_sets:
        print(f"* SPLIT {split}")

        scene_info = np.load(
            os.path.join(data_dir, split, "scene_info.npy"), allow_pickle=True
        ).item()
        scene_cams = scene_info["cams"]
        scene_frames = scene_info["frames"]

        frame_obj_info = np.load(
            os.path.join(data_dir, split, "frame_obj_anno.npy"), allow_pickle=True
        ).item()
        matching_frames_info = np.load(
            os.path.join(data_dir, split, "matching_frames.npy"), allow_pickle=True
        )

        with open(os.path.join(data_dir, split, "frame_rgb_names.pickle"), "rb") as f:
            rgb_frame_names = pickle.load(f)

        scene_name_list = [scene_name for scene_name in scene_frames.keys()]

        unique_ids = list(
            set([tuple(m[:3].astype(np.int32)) for m in matching_frames_info])
        )
        if SUBSET_ONLY:
            unique_ids = [m for m in unique_ids if m[0] == 0]

        assignments[split] = {}
        th_list = [np.round(i * (5.0 / 100), decimals=2) for i in range(20)]
        tmp_found_objects_per_scene = {}
        tmp_gt_objects_per_scene = {}
        tp, total_n_obj_pred, total_n_obj_gt = {}, {}, {}
        for th in th_list:
            tmp_found_objects_per_scene[th] = {}
            tmp_gt_objects_per_scene[th] = {}
            for scene_name in scene_name_list:
                tmp_found_objects_per_scene[th][scene_name] = set()
                tmp_gt_objects_per_scene[th][scene_name] = set()
            tp[th] = 0
            total_n_obj_pred[th] = 0
            total_n_obj_gt[th] = 0

        for id in unique_ids:
            scene_id, cam_id, frame_id = id
            scene_name = scene_name_list[scene_id]
            cams = scene_cams[scene_name]
            cam_name = cams[cam_id]
            frame_name = rgb_frame_names[scene_id][cam_id][frame_id]
            # print(scene_name, cam_name, frame_name)

            if scene_name not in assignments[split].keys():
                assignments[split][scene_name] = {}
            if cam_name not in assignments[split][scene_name].keys():
                assignments[split][scene_name][cam_name] = {}
            if frame_name in assignments[split][scene_name][cam_name].keys():
                print(
                    f"ERROR: frame {scene_name}/{cam_name}/{frame_name} multiple times in list"
                )

            cur_output_file = os.path.join(
                output_dir, "detections", scene_name, cam_name, frame_name + "pkl"
            )
            with open(cur_output_file, "rb") as f:
                detections = pickle.load(f)

            pred_scores = detections["detection_scores"]

            for th in th_list:
                score_filter = pred_scores >= th
                if NYU40_CATEGORY_INDEX is not None:
                    pred_cls = detections["detection_classes"][score_filter]
                else:
                    _, pred_cls = map_coco2nyu_lbls(
                        detections["detection_classes"][score_filter]
                    )
                pred_bb = detections["detection_boxes"][score_filter]
                pred_bb = np.asarray(
                    [
                        [
                            int(bb[0] * IMAGE_HEIGHT),
                            int(bb[1] * IMAGE_WIDTH),
                            int(bb[2] * IMAGE_HEIGHT),
                            int(bb[3] * IMAGE_WIDTH),
                        ]
                        for bb in pred_bb
                    ]
                )

                # --> filtering (e.g. classes, too small detections) can be done during data loading

                gt_cls = frame_obj_info["cls"][scene_id][cam_id][frame_id]
                gt_bb = frame_obj_info["bb2d"][scene_id][cam_id][frame_id]
                gt_obj_idx = frame_obj_info["id"][scene_id][cam_id][frame_id]

                n_obj_gt = len(gt_cls)
                n_obj_pred = pred_bb.shape[0]

                bb_length = np.asarray([[o[2] - o[0], o[3] - o[1]] for o in gt_bb])
                bb_size = bb_length[:, 0] * bb_length[:, 1]

                # get matches based on IoU
                bb_ious = np.zeros((n_obj_pred, n_obj_gt))
                for i in range(n_obj_pred):
                    for j in range(n_obj_gt):
                        bb_ious[i, j] = get_iou(pred_bb[i], gt_bb[j])

                bb_ious_th = bb_ious >= 0.5

                bb_ious_max_wrt_pred = np.zeros_like(bb_ious).astype(bool)
                bb_ious_argmax_gt = np.argmax(bb_ious, axis=1)
                for i in range(n_obj_pred):
                    bb_ious_max_wrt_pred[i, bb_ious_argmax_gt[i]] = True
                bb_ious_max_wrt_gt = np.zeros_like(bb_ious).astype(bool)
                if (
                    n_obj_pred > 0
                ):  # otherwise, no object has been detected (would yield runtime error for the following)
                    bb_ious_argmax_pred = np.argmax(bb_ious, axis=0)
                    for j in range(n_obj_gt):
                        bb_ious_max_wrt_gt[bb_ious_argmax_pred[j], j] = True
                bb_ious_max = bb_ious_max_wrt_gt * bb_ious_max_wrt_pred * bb_ious_th

                detected_pred = np.sum(bb_ious_max, axis=1)
                wrong_pred = np.ones_like(detected_pred) - detected_pred
                if n_obj_pred > 0 and detected_pred.max() > 1:
                    print(
                        "[WARNING] Multiple matches for predicted object",
                        id,
                        detected_pred.max(),
                    )
                    # print(bb_ious_max)
                # TMP: iterate over all matched gt objects;
                #      -> test whether there are multiple predicted objects matched to the same gt objects
                detected_gt = np.sum(bb_ious_max, axis=0)
                if detected_gt.max() > 1:
                    print(
                        "[WARNING] Multiple matches for gt object",
                        id,
                        detected_pred.max(),
                    )
                    # print(bb_ious_max)

                assign_id = np.where(
                    detected_pred,
                    bb_ious_argmax_gt,
                    -1 * np.ones_like(bb_ious_argmax_gt, dtype=np.int32),
                )
                assign_cls = []
                assign_bb = []
                for i in range(n_obj_pred):
                    assign_cls.append(pred_cls[i])
                    assign_bb.append(pred_bb[i])

                tp[th] += np.sum(detected_pred)
                total_n_obj_pred[th] += n_obj_pred
                total_n_obj_gt[th] += n_obj_gt

                for i in range(len(gt_obj_idx)):
                    tmp_gt_objects_per_scene[th][scene_name].add(gt_obj_idx[i])
                for i in range(len(assign_id)):
                    if assign_id[i] >= 0:
                        tmp_found_objects_per_scene[th][scene_name].add(
                            gt_obj_idx[assign_id[i]]
                        )

                if th == MIN_SCORE_THRESHOLD:

                    assignments[split][scene_name][cam_name][frame_name] = {
                        "gt_obj_id": assign_id,
                        "pred_obj_cls": np.asarray(assign_cls),
                        "pred_obj_bb": np.asarray(assign_bb),
                    }

                    # get statistics
                    for j in range(n_obj_gt):
                        if gt_cls[j] in classes_ids_intersect:
                            cls_gt_new = classes_ids_intersect.index(gt_cls[j])
                        else:
                            cls_gt_new = classes_ids_intersect.index(
                                39
                            )  # -> other property
                        sz = bb_size[j] / (768 * 1024)  # relative bb size
                        if detected_gt[j] == 1:
                            detected_classes[cls_gt_new] += 1
                            detected_sizes.append(sz)
                            detect_total_number_per_image_per_object.append(n_obj_gt)
                        else:
                            missing_classes[cls_gt_new] += 1
                            missing_sizes.append(sz)
                            missing_total_number_per_image_per_object.append(n_obj_gt)

                    n_pred_assigned = np.sum(detected_pred)
                    n_pred_wrong = np.sum(wrong_pred)
                    n_gt_missing = n_obj_gt - n_pred_assigned
                    detect_total_number_per_image.append(n_pred_assigned)
                    missing_total_number_per_image.append(n_gt_missing)
                    wrong_total_number_per_image.append(n_pred_wrong)
                    total_number_per_image.append(n_obj_gt)

        n_somewhere_found_objects = {}
        n_fully_missed_objects = {}
        ratio_global_found = {}
        prec = {}
        rec = {}
        for th in th_list:
            n_somewhere_found_objects[th] = 0
            n_fully_missed_objects[th] = 0
            for scene_name in scene_name_list:
                missed_obj = (
                    tmp_gt_objects_per_scene[th][scene_name]
                    - tmp_found_objects_per_scene[th][scene_name]
                )
                n_fully_missed_objects[th] += len(missed_obj)
                n_somewhere_found_objects[th] += len(
                    tmp_found_objects_per_scene[th][scene_name]
                )
            ratio_global_found[th] = n_somewhere_found_objects[th] / (
                n_somewhere_found_objects[th] + n_fully_missed_objects[th]
            )
            prec[th] = tp[th] / (total_n_obj_pred[th] + 10**-5)
            rec[th] = tp[th] / total_n_obj_gt[th]

        LOG_FILE.write(f"  -> Split: {split}, Th={MIN_SCORE_THRESHOLD}")
        LOG_FILE.write(
            f" \t\t tp={tp[MIN_SCORE_THRESHOLD]}, "
            f"#obj(detect)={total_n_obj_pred[MIN_SCORE_THRESHOLD]}, "
            f"#obj(gt)={total_n_obj_gt[MIN_SCORE_THRESHOLD]}, "
            f"#somewhere_found_objects={n_somewhere_found_objects[MIN_SCORE_THRESHOLD]}, "
            f"#fully_missed_obj={n_fully_missed_objects[MIN_SCORE_THRESHOLD]}"
        )
        LOG_FILE.write(
            f" \t\t prec={prec[MIN_SCORE_THRESHOLD]:.3f}, rec={rec[MIN_SCORE_THRESHOLD]:.3f}, "
            f"rat={ratio_global_found[MIN_SCORE_THRESHOLD]:.3f}"
        )

        # --- Statistics over multiple thresholds
        ratio_global_found_list = [ratio_global_found[th] for th in th_list]
        prec_list = [prec[th] for th in th_list]
        rec_list = [rec[th] for th in th_list]
        viz.plot_xy_curve(
            th_list,
            ratio_global_found_list,
            "th",
            "ratio_global_found",
            "rat_global_found_" + split,
            path=output_dir,
        )
        viz.plot_xy_curve(
            rec_list, prec_list, "rec", "prec", "prec_rec_" + split, path=output_dir
        )

        # --- Statistics wrt. single, fixed threshold
        # Plot, wrt. Classification
        detected_classes_abs_dict = {}
        missing_classes_abs_dict = {}
        detected_classes_rel_dict = {}
        missing_classes_rel_dict = {}
        for cls_id, cls_name in enumerate(classes_names_intersect):
            detected_classes_abs_dict[cls_name] = detected_classes[cls_id]
            missing_classes_abs_dict[cls_name] = missing_classes[cls_id]

            n_total_class = detected_classes[cls_id] + missing_classes[cls_id]
            detected_classes_rel_dict[cls_name] = (
                detected_classes[cls_id] / n_total_class
            )
            missing_classes_rel_dict[cls_name] = missing_classes[cls_id] / n_total_class
        viz.plot_bars(
            [detected_classes_abs_dict, missing_classes_abs_dict],
            ["detected", "missing"],
            "Detections/ Missing by object category",
            "obj-cls_detections_" + split + "_abs",
            path=output_dir,
        )
        viz.plot_bars(
            [detected_classes_rel_dict, missing_classes_rel_dict],
            ["detected", "missing"],
            "Detections/ Missing by object category",
            "obj-cls_detections_" + split + "_rel",
            path=output_dir,
        )

        # Plot, wrt. Size
        viz.plot_histogramm(
            [detected_sizes, missing_sizes],
            ["detected", "missing"],
            "Detections/ Missing wrt. bb size",
            bins=None,
            path=os.path.join(output_dir, "obj-size_detections_" + split + "_abs"),
        )

        # Plot wrt. object count (per image)
        viz.plot_histogramm(
            [
                detect_total_number_per_image,
                wrong_total_number_per_image,
                missing_total_number_per_image,
                total_number_per_image,
            ],
            ["detected", "wrong", "missing", "total"],
            "Detections/ Missing full #objects in image",
            bins=None,
            path=os.path.join(output_dir, "obj-count_detections_ " + split + "_abs"),
        )
        viz.plot_histogramm(
            [
                detect_total_number_per_image_per_object,
                missing_total_number_per_image_per_object,
            ],
            ["detected", "missing"],
            "Detections/ Missing per object wrt. #objects in image",
            bins=None,
            path=os.path.join(output_dir, "obj-count_detections_v2_" + split + "_abs"),
        )

    # save matches (predicted detections, gt)
    output_file = os.path.join(output_dir, "assignment.npy")
    np.save(output_file, assignments)


# --------------------------------------------------


if __name__ == "__main__":

    FLAGS = parse_arguments()

    RUN_ON_CLUSTER = FLAGS.run_on_cluster
    SUBSET_ONLY = FLAGS.subset_only
    CHECK_MISSING_ONLY = FLAGS.check_missing_only

    if not os.path.exists(FLAGS.data_dir):
        print(f"Data path {FLAGS.data_dir} does not exist.")
        exit(0)

    output_dir = os.path.join(FLAGS.data_dir, "Hypersim_Detections_" + FLAGS.model)
    if SUBSET_ONLY:
        output_dir += "_subset"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Get Model
    #   -> download should be done before
    if FLAGS.model == "EfficientDet-D7":
        model_name = "efficientdet_d7_coco17_tpu-32"
        base_dir = PATH_TO_KERAS_DATA
    else:
        print(f"[ERROR] Model {FLAGS.model} unknown/ not downloaded yet.")
        exit()
    path_to_saved_model = os.path.join(base_dir, model_name, "saved_model")

    # Load Model
    print("Loading model...", end="")
    start_time = time.time()

    detect_fn = tf.saved_model.load(path_to_saved_model)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done! Took {} seconds".format(elapsed_time))

    # Load label map data (for plotting)
    path_to_labels = os.path.join(PATH_TO_KERAS_DATA, "mscoco_label_map.pbtxt")
    COCO_CATEGORY_INDEX = label_map_util.create_category_index_from_labelmap(
        path_to_labels, use_display_name=True
    )

    if not RUN_ON_CLUSTER:
        from utils import viz
        from utils.helper_funcs import *

        LOG_FILE = LogFile(os.path.join(output_dir, "log_eval.txt"))
        LOG_FILE.write("{:%d.%m_%H:%M}".format(datetime.now()))

    # Run..
    get_detections(detect_fn, output_dir)

    preprocessed_data_dir = os.path.join(FLAGS.data_dir, "Hypersim_Processed_ICRA23")
    if SUBSET_ONLY:
        preprocessed_data_dir += "_subset"

    if not RUN_ON_CLUSTER and not CHECK_MISSING_ONLY:
        eval_detections(preprocessed_data_dir, output_dir)

    exit(0)
