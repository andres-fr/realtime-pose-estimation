# -*- coding:utf-8 -*-


"""
Version of validate_inference trimmed down to perform inference
to find heads.

ffmpeg -i 0481BL.MXF 0481BL.MXF_%03d.png
"""

import surgery
import caffe

import os
#
import matplotlib
import numpy as np
from PIL import Image
import torch
import torchvision
#
from rtpe.third_party.transforms import resize_align_multi_scale
from rtpe.third_party.group import HeatmapParser
from rtpe.third_party.vis import save_valid_image
#
from rtpe.dataloaders import CocoDistillationDatasetAugmented
from rtpe.helpers import get_hrnet_w48_teacher
from rtpe.helpers import plot_arrays


# #############################################################################
# # HELPERS
# #############################################################################
def normalize_uint8_arr(arr):
    """
    """
    mn, mx = arr.min(), arr.max()
    if mn != mx:
        arr32 = np.float32(arr)
        arr32 -= mn
        arr32 *= 255.0 / mx
        return arr32.astype(np.uint8)


def groups_to_heads(groups, p_thresh=0.3):
    """
    """
    heads = [np.float32([(x, y) for x, y, p in g[0][:4, :3] if p>=p_thresh])
             for g in groups]
    return heads


def expand_bbox(x0, x1, y0, y1, expansion_ratio=1.5, clip_wh=None):
    """
    Without going over the boundaries
    """
    center_x, center_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    w, h = x1 - x0, y1 - y0
    w_exp_half, h_exp_half = (expansion_ratio * w* 0.5), (expansion_ratio * h * 0.5)
    #
    exp_bbox = (center_x - w_exp_half, center_x + w_exp_half,
                center_y - h_exp_half, center_y + h_exp_half)
    if clip_wh is not None:
        exp_bbox = (max(0, round(exp_bbox[0])),
                    min(clip_wh[0], round(exp_bbox[1])),
                    max(0, round(exp_bbox[2])),
                    min(clip_wh[1], round(exp_bbox[3])))
    #
    return exp_bbox



# #############################################################################
# # GLOBALS
# #############################################################################
DEVICE = "cuda"
HOME = os.path.expanduser("~")
#
MODEL_PATH = "models/pose_higher_hrnet_w48_640.pth.tar"
NUM_HEATMAPS = 17
INPUT_SIZE = 640  # this is hardcoded to the architecture
HM_PARSER_PARAMS = {"max_num_people": 30,
                    "detection_threshold": 0.1,
                    "tag_threshold": 1.0,
                    "use_detection_val": True,
                    "ignore_too_much": False,
                    "tag_per_joint": True,
                    "nms_ksize": 5,
                    "nms_padding": 2}
#
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STDDEV = [0.229, 0.224, 0.225]
VAL_GT_STDDEVS = [2.0]
#
# SAVE_DIR = "/tmp"
PLOT = False
KP_THRESH = 0.0001
BBOX_RADIUS = 90 # (pixels)
IMG_DIR = "/shared/mvn1e/mocap_library/mairi_png"

# #############################################################################
# # MAIN ROUTINE
# #############################################################################
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net("../face_segmentation/data/face_seg_fcn8s_deploy.prototxt",
                "../face_segmentation/data/face_seg_fcn8s.caffemodel",
                caffe.TEST)


# dataloaders
IMG_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=IMG_NORM_MEAN,
                                     std=IMG_NORM_STDDEV,
                                     inplace=True)])

img_paths = [os.path.join(IMG_DIR, p) for p in os.listdir(IMG_DIR) if p.endswith(".png")]


# model
hhrnet = get_hrnet_w48_teacher(MODEL_PATH).to(DEVICE)
hhrnet.eval()

hm_parser = HeatmapParser(num_joints=NUM_HEATMAPS,
                          **HM_PARSER_PARAMS)

# main loop
all_preds = []
all_scores = []
for ii, imgpath in enumerate(img_paths):
    img = Image.open(imgpath).convert("RGB")
    print(ii, "processing", imgpath, img.size)
    resized_img, center, scale = resize_align_multi_scale(np.array(img),
                                                          INPUT_SIZE, 1, 1)
    t = IMG_TRANSFORM(resized_img).unsqueeze(0).to(DEVICE)
    w, h = img.size
    # run HHRNet inference
    with torch.no_grad():
        preds, refined = hhrnet(t)
        hms = torch.nn.functional.interpolate(
            refined, (h, w), mode="bilinear", align_corners=True)
        aes = torch.nn.functional.interpolate(
            preds[:, NUM_HEATMAPS:, :, :], (h, w),
            mode="bilinear", align_corners=True)
    # parser accepts hms(1, 17, h, w) and ae (1, AE_DIM, h, w, 1)
    grouped, scores = hm_parser.parse(hms, aes.unsqueeze(-1),
                                      adjust=True, refine=True)
    grouped = [g for g in grouped if len(g>0)]
    if not grouped:
        continue
    print(grouped)
    heads = groups_to_heads(grouped, KP_THRESH)
    head_bboxes = []
    for h in heads:
        min_x, max_x = h[:, 0].min(), h[:, 0].max()
        min_y, max_y = h[:, 1].min(), h[:, 1].max()
        head_bboxes.append((min_x, max_x, min_y, max_y))


    # for each bbox, obtain center and max radius, and redefine bboxes
    head_centers = [(0.5*(x0 + x1), 0.5*(y0 + y1)) for x0, x1, y0, y1 in head_bboxes]
    # HARDCODED RADIUS:
    # head_maxrads = [0.5 * max(x1 - x0, y1 - y0) for x0, x1, y0, y1 in head_bboxes]
    head_maxrads = [BBOX_RADIUS for x0, x1, y0, y1 in head_bboxes]
    expanded_bboxes = [(hc_x - rad, hc_y - rad, hc_x + rad, hc_y + rad) for (hc_x, hc_y), rad in zip(head_centers, head_maxrads)]
    print(imgpath, head_centers, expanded_bboxes)
    # expanded_bboxes = [(x0, x1, y0, y1) for x0, y0, x1, y1 in expanded_bboxes]
    expanded_bboxes = [expand_bbox(x0, x1, y0, y1,
                                   expansion_ratio=1, clip_wh=img.size)
                       for x0, y0, x1, y1 in expanded_bboxes]

    # print(">>>>", len(grouped), grouped)
    # import pdb; pdb.set_trace()
    # for evaluation
    final_results = [x for x in grouped[0] if x.size > 0]
    all_preds.append(final_results)
    all_scores.append(scores)
    # # save predictions
    # save_valid_image(np.array(img), [x for x in grouped[0] if x.size > 0],
    #                  os.path.join(SAVE_DIR,
    #                               "validate_hhrnet_{}.jpg".format(ii)),
    #                  dataset="COCO")    # plot predictions

    arr = np.array(img)
    face_mask = np.zeros((arr.shape[0], arr.shape[1])).astype(np.bool)
    head_arrs = [arr[int(y0):int(y1), int(x0):int(x1)]
                 for x0, x1, y0, y1 in expanded_bboxes]
    #
    for ha, (x0, x1, y0, y1)  in zip(head_arrs, expanded_bboxes):
        norm = normalize_uint8_arr(ha)
        ha_im = Image.fromarray(norm)
        ha_im = ha_im.resize((400, 400))
        in_ = np.array(ha_im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_ = in_.transpose((2,0,1))
        #
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        net.forward()
        out = net.blobs['score'].data[0].argmax(axis=0) !=0
        #
        out_patch = np.array(Image.fromarray(out).resize((ha.shape[1], ha.shape[0])))
        yyy, xxx = np.where(out_patch!=0)
        #
        face_mask[y0 + yyy, x0 + xxx] = True
        # plot_arrays(arr, face_mask)
        plot_arrays(arr, arr + 50*face_mask[:, :, None])
        # plot_arrays(ha_im, np.array(ha_im)*mask[:, :, None],
        #             np.array(ha_im) * (1-mask[:, :, None]))


    if PLOT:
        matplotlib.use("TkAgg")
        # import pdb; pdb.set_trace()
        plot_arrays(img, *head_arrs, share_zoom=False)
