# -*- coding:utf-8 -*-


"""
This module first runs a dataloader with data augmentation, optionally plotting
the augmented data, and then performs a minival run, saving the stem outputs to
a given location as images.
"""


import os
#
import torch
import torchvision
import matplotlib
#
from rtpe.third_party.fp16_utils.fp16util import network_to_half
#
from rtpe.helpers import SeededCompose, plot_arrays
from rtpe.dataloaders import CocoDistillationDatasetAugmented
from rtpe.students import StemHRNet

# #############################################################################
# # GLOBALS
# #############################################################################
OUTPUT_DIR = "/tmp"

HOME = os.path.expanduser("~")
COCO_DIR = os.path.join(HOME, "datasets", "coco")
MINIVAL_FILE = "assets/coco_minival2017_100.txt"
MODEL_PATH = "models/pose_higher_hrnet_w48_640.pth.tar"

with open(MINIVAL_FILE, "r") as f:
    MINIVAL_IDS = [int(line.rstrip('.jpg\n')) for line in f]

IMG_NORMALIZE_TRANSFORM = torchvision.transforms.Compose([
    # jitter? to gray?
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225],
                                     inplace=True)
])

OVERALL_HHRNET_TRANSFORM = SeededCompose([
    torchvision.transforms.ToPILImage(mode="F"),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomAffine(
        degrees=(-45, +45), translate=(0.1, 0.1), scale=(0.7, 1.3)),
    torchvision.transforms.RandomCrop(size=(480, 480), pad_if_needed=True),
    torchvision.transforms.ToTensor()])


# #############################################################################
# # MAIN ROUTINE
# #############################################################################
minival_dataset = CocoDistillationDatasetAugmented(
    COCO_DIR, "val2017",
    os.path.join(COCO_DIR, "hrnet_predictions", "val2017"),
    gt_stddevs_pix=[2.0],
    img_transform=IMG_NORMALIZE_TRANSFORM,
    whitelist_ids=MINIVAL_IDS)

val_augm_dataset = CocoDistillationDatasetAugmented(
    COCO_DIR, "val2017",
    os.path.join(COCO_DIR, "hrnet_predictions", "val2017"),
    gt_stddevs_pix=[20.0, 9.0, 2.0],
    img_transform=IMG_NORMALIZE_TRANSFORM,
    overall_transform=OVERALL_HHRNET_TRANSFORM)


stem = network_to_half(StemHRNet())
stem[1].load_pretrained(MODEL_PATH, device="cuda")
stem.to("cuda")


i = 1
print("TRAIN >>>", i)
img_id, img, mask, hms, teach_hms, teach_ae, segm_mask = val_augm_dataset[i]
with torch.no_grad():
    stem_out = stem(img.unsqueeze(0).to("cuda")).to("cpu").squeeze()
    matplotlib.use("TkAgg")
    plot_arrays(img.permute(1, 2, 0), mask, *[hm.max(dim=0)[0] for hm in hms],
                teach_hms.sum(dim=0))
    for i_p, plane in enumerate(stem_out):
        outpath = os.path.join(OUTPUT_DIR, "id{}_plane{}.png".format(i, i_p))
        torchvision.utils.save_image(plane, outpath, normalize=True)


breakpoint()
