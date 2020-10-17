# -*- coding:utf-8 -*-


"""
This module is a simplified version of the HHRNet inference and validation from
https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation

It can be used to investigate different behaviours of the original model (like
e.g. reducing AE dimensionality) and to plot/export the predictions.
"""


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
# # GLOBALS
# #############################################################################
DEVICE = "cuda"
HOME = os.path.expanduser("~")
COCO_DIR = os.path.join(HOME, "datasets", "coco")
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
PLOT_EVERY = 100
SAVE_EVERY = 100  # None
SAVE_DIR = "/tmp"


# #############################################################################
# # MAIN ROUTINE
# #############################################################################

# dataloaders
IMG_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=IMG_NORM_MEAN,
                                     std=IMG_NORM_STDDEV,
                                     inplace=True)])

val_ds = CocoDistillationDatasetAugmented(
    COCO_DIR, "val2017", remove_images_without_annotations=False)

img_paths = [os.path.join(COCO_DIR, "images", "val2017",
                          "{:012d}.jpg".format(x)) for x in val_ds.ids]

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
    # for evaluation
    final_results = [x for x in grouped[0] if x.size > 0]
    all_preds.append(final_results)
    all_scores.append(scores)
    # save predictions
    if SAVE_EVERY is not None and ii % SAVE_EVERY == 0:
        save_valid_image(np.array(img), [x for x in grouped[0] if x.size > 0],
                         os.path.join(SAVE_DIR,
                                      "validate_hhrnet_{}.jpg".format(ii)),
                         dataset="COCO")    # plot predictions
    if PLOT_EVERY is not None and ii % PLOT_EVERY == 0:
        matplotlib.use("TkAgg")
        plot_arrays(img, hms[0].sum(dim=0).cpu(), aes[0].mean(dim=0).cpu())
#
eval_dict, mAP = val_ds.evaluate(all_preds, all_scores, ".", False, False)
breakpoint()
eval_str = "\n".join([k+"="+str(v) for k, v in eval_dict.items()])
print(eval_str)
