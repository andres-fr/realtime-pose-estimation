# -*- coding:utf-8 -*-


"""
This module first runs a dataloader with data augmentation, optionally plotting
the augmented data, and then performs a minival run, also optionally plotting.
"""


import os
#
import torchvision
import matplotlib
#
from rtpe.third_party.group import HeatmapParser
from rtpe.third_party.vis import save_valid_image
#
from rtpe.helpers import SeededCompose, plot_arrays
from rtpe.dataloaders import CocoDistillationDatasetAugmented


# #############################################################################
# # GLOBALS
# #############################################################################
NUM_TRAIN_PLOTS = 2
NUM_MINIVAL_PLOTS = 2
MINIVAL_OUTPUT_DIR = "/tmp"

HOME = os.path.expanduser("~")
COCO_DIR = os.path.join(HOME, "datasets", "coco")
MINIVAL_FILE = "assets/coco_minival2017_100.txt"


with open(MINIVAL_FILE, "r") as f:
    MINIVAL_IDS = [int(line.rstrip('.jpg\n')) for line in f]

IMG_NORMALIZE_TRANSFORM = torchvision.transforms.Compose([
    # jitter? to gray?
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225],
                                     inplace=True)])

OVERALL_HHRNET_TRANSFORM = SeededCompose([
    torchvision.transforms.ToPILImage(mode="F"),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomAffine(
        degrees=(-45, +45), translate=(0.1, 0.1), scale=(0.7, 1.3)),
    # torchvision.transforms.RandomCrop(size=(480, 480), pad_if_needed=True),
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

hm_parser = HeatmapParser(num_joints=17,
                          max_num_people=30,
                          detection_threshold=0.1,
                          tag_threshold=1.0,
                          use_detection_val=True,
                          ignore_too_much=False,
                          tag_per_joint=True,
                          nms_ksize=5,
                          nms_padding=2)


# "TRAINING" SET
for i in range(NUM_TRAIN_PLOTS):
    print("TRAIN >>>", i)
    img_id, img, mask, hms, teach_hms, teach_ae = val_augm_dataset[i]
    # plot: img, mask, ground truths, teacher detection
    matplotlib.use('TkAgg')
    plot_arrays(img.permute(1, 2, 0), mask, *[hm.max(dim=0)[0] for hm in hms],
                teach_hms.sum(dim=0))


# MINIVAL DATASET
all_preds = []
all_scores = []
len_minival_dataset = len(minival_dataset)
for i in range(len_minival_dataset):
    print("MINIVAL >>>", i)
    img_id, img, mask, hms, teach_hms, teach_ae = minival_dataset[i]
    grouped, scores = hm_parser.parse(
        teach_hms.unsqueeze(0), teach_ae.unsqueeze(0),
        adjust=True, refine=True)
    # for evaluation
    final_results = [x for x in grouped[0] if x.size > 0]
    all_preds.append(final_results)
    all_scores.append(scores)
    # for visualization
    if NUM_MINIVAL_PLOTS > 0 and i % (len_minival_dataset//NUM_MINIVAL_PLOTS) == 0:
        # save teacher detections to path as images
        save_valid_image(
            img.sub(img.min()).mul(255.0 / img.max()).cpu().permute(
                1, 2, 0).numpy(),
            [x for x in grouped[0] if x.size > 0],
            os.path.join(MINIVAL_OUTPUT_DIR, "hhrnet_minival_{}.jpg".format(i)),
            dataset="COCO")
        # plot: img, ground truths, teacher detection
        plot_arrays(img.permute(1, 2, 0), *[hm.max(dim=0)[0] for hm in hms],
                    teach_hms.sum(dim=0))


# OKS evaluation on minival dataset
eval_dict, mAP = minival_dataset.evaluate(all_preds, all_scores, ".", False, False)
eval_str = "\n".join([k+"="+str(v) for k, v in eval_dict.items()])
print(eval_str)
breakpoint()
