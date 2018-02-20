import os
import argparse
import pickle
import numpy as np
from scipy import misc
import cv2

import pycocotools.mask as mask_util

import datasets.dummy_datasets as dummy_datasets
import utils.vis as vis_utils
import utils_ade20k.misc as ade20k_utils


def vis(im_name, im, cls_boxes, cls_segms, cls_keyps):
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    out_dir = None

    loaded = vis_utils.vis_one_image_opencv(im, cls_boxes, segms=cls_segms, keypoints=cls_keyps, thresh=0.9, kp_thresh=2,
        show_box=False, dataset=None, show_class=False)
    misc.imsave("loaded.png", loaded)

def create_panoptic_segmentation(cls_boxes, cls_segms, cls_keyps):
    boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)
    dataset = dummy_datasets.get_coco_dataset()

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    segm_out = np.zeros(im.shape[:2], dtype="uint8")
    inst_out = np.zeros(im.shape[:2], dtype="uint8")

    masks = mask_util.decode(segms)
    cnt = 1
    for i in sorted_inds:
        mask = masks[...,i]
        mask = np.nonzero(mask)
        class_name = dataset.classes[classes[i]]
        idx = ade20k_utils.category_to_idx(class_name)

        segm_out[mask] = idx
        inst_out[mask] = cnt
        cnt += 1

    misc.imsave("test.png", segm_out)
    out = np.stack([segm_out, inst_out], axis=-1)
    return out


def process(im_name, im, pkl_path):
    cls_boxes, cls_segms, cls_keyps = pickle.load(open(pkl_path, 'rb'))
    out = create_panoptic_segmentation(cls_boxes, cls_segms, cls_keyps)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, required=True, help="Project name")
    args = parser.parse_args()

    config = utils_ade20k.get_config(args.project)

    im_name = "hi"
    im = misc.imread(im_path, mode='RGB')
    process(im_name, im, pkl_path)