# -*- coding:utf-8 -*-

"""
This module uses the ``higher_hrnet_w48_640`` model distributed by
https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
to perform heatmap and associative embedding inference on a given set of
images, and save them as compressed numpy arrays. Usage example::

  python teacher_inference.py  -I ~/datasets/coco/images/val2017/*
  -o /tmp -m models/pose_higher_hrnet_w48_640.pth.tar
"""


import os
import torch
import torch.backends.cudnn as cudnn
import torchvision
from PIL import Image
import numpy as np
import argparse
#
from rtpe.helpers import get_hrnet_w48_teacher
from rtpe.third_party.transforms import resize_align_multi_scale
from rtpe.third_party.group import HeatmapParser
from rtpe.third_party.vis import save_valid_image

# #############################################################################
# # GLOBALS
# #############################################################################

# model config
NUM_HEATMAPS = 17
HM_PARSER_PARAMS = {"max_num_people": 30,
                    "detection_threshold": 0.1,
                    "tag_threshold": 1.0,
                    "use_detection_val": True,
                    "ignore_too_much": False,
                    "tag_per_joint": True,
                    "nms_ksize": 5,
                    "nms_padding": 2}

# backend config
cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

# these are hardcoded to the distributed w48 model
INPUT_SIZE = 640
HALF_PRECISION = True
HEATMAPS_ORDER = ["nose", "leye", "reye", "lear", "rear", "lshould", "rshould",
                  "lelbow", "relbow", "lwrist", "rwrist", "lhip", "rhip",
                  "lknee", "rknee", "lankle", "rankle"]


# #############################################################################
# # GLOBALS
# #############################################################################
parser = argparse.ArgumentParser("HigherHRNet Inference")
parser.add_argument("-I", "--input_paths", required=True, type=str, nargs="+",
                    help="Abs paths for the input images")
parser.add_argument("-o", "--out_dir", required=True, type=str,
                    help="Path to output the predictions")
parser.add_argument("-m", "--model_path", required=True, type=str,
                    help="Path to the HigherHRNet_w48_640 state dict")
parser.add_argument("-C", "--force_cpu", action="store_true",
                    help="Run on CPU even if CUDA is present")
args = parser.parse_args()
#
IMG_PATHS = args.input_paths
# xxx = "/home/shared/datasets/coco/train2017/"
# IMG_PATHS = [os.path.join(xxx, p) for p in os.listdir(xxx)]
OUT_DIR = args.out_dir
MODEL_PATH = args.model_path
DEVICE = "cuda" if (not args.force_cpu and
                    torch.cuda.is_available()) else "cpu"

model = get_hrnet_w48_teacher(MODEL_PATH).to(DEVICE)
hm_parser = HeatmapParser(num_joints=NUM_HEATMAPS,
                          **HM_PARSER_PARAMS)

for img_path in IMG_PATHS:
    out_path = os.path.join(OUT_DIR,
                            os.path.basename(img_path)) + "_w48_predictions"
    preproc_img = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
    #
    img = Image.open(img_path).convert("RGB")  # gray imgs -> 3 chans
    print("processing", img_path, img.size)
    resized_img, center, scale = resize_align_multi_scale(np.array(img),
                                                          INPUT_SIZE, 1, 1)
    t = preproc_img(resized_img).unsqueeze(0).to(DEVICE)
    w, h = img.size
    # run HHRNet inference
    with torch.no_grad():
        preds, refined = model(t)
        hms = torch.nn.functional.interpolate(
            refined, (h, w), mode="bilinear", align_corners=True)
        aes = torch.nn.functional.interpolate(
            preds[:, NUM_HEATMAPS:, :, :], (h, w),
            mode="bilinear", align_corners=True)
    # parser accepts hms(1, 17, h, w) and ae (1, AE_DIM, h, w, 1)
    grouped, scores = hm_parser.parse(hms, aes.unsqueeze(-1),
                                      adjust=True, refine=True)
    # Saving keypoints, scores and rendered image
    np.savez_compressed(out_path, keypoints=grouped, scores=scores,
                        heatmaps_order=HEATMAPS_ORDER)
    save_valid_image(np.array(img), [x for x in grouped[0] if x.size > 0],
                     out_path + "_rendered.jpg", dataset="COCO")

# import matplotlib.pyplot as plt
# # # plot img
# # plt.clf(); plt.imshow(resized_img); plt.show()
# # # plot all heatmaps
# # plt.clf(); plt.imshow(hms.squeeze().cpu().sum(0)); plt.show()

# # plot separate heatmaps on the img
# p = torch.nn.functional.interpolate(heatmaps, size=t[0, 0].shape,
#                                     mode="bilinear").squeeze() * 10
# p += t[0, 0]
# breakpoint()
# for pp in p:
#     plt.clf();  plt.imshow(pp.cpu());  plt.show()
