# -*- coding:utf-8 -*-


"""
This module is a small utility script that instantiates the model, preloads the
pretrained attention parameters and runs the validation dataset, plotting the
attention maps and computing the loss.
"""

import os
import time
#
import torch
import torch.backends.cudnn as cudnn
import torchvision
from tensorboardX import SummaryWriter
import matplotlib
#
from rtpe.third_party.group import HeatmapParser
#
from rtpe.helpers import SeededCompose, make_timestamp, ColorLogger, \
    ModuleSummary, plot_arrays
from rtpe.dataloaders import CocoDistillationDatasetAugmented2
from rtpe.students import AttentionStudent, AttentionStudentSteps
# from rtpe.optimization import get_sgd_optimizer, SgdrScheduler, \
#     DistillationBceLossKeypointMining
from rtpe.engine import eval_student




# #############################################################################
# # GLOBALS
# #############################################################################
PLOT_EVERY = 1


EASY_VAL_BIG_PATH = "assets/coco_val_easy_big.txt"
EASY_VAL_MED_PATH = "assets/coco_val_easy_med.txt"
EASY_VAL_SMALL_PATH = "assets/coco_val_easy_small.txt"

# general/pathing
DEVICE = "cuda"
HOME = os.path.expanduser("~")
COCO_DIR = os.path.join(HOME, "datasets", "coco")
HRNET_TRAIN_DIR = os.path.join(COCO_DIR, "hrnet_predictions", "train2017")
HRNET_VAL_DIR = os.path.join(COCO_DIR, "hrnet_predictions", "val2017")

# model arch
NUM_HEATMAPS = 17
AE_DIMENSIONS = 1
MODEL_PATH = "models/pose_higher_hrnet_w48_640.pth.tar"
LAYERS_PER_STAGE = [2, 2]
HALF_PRECISION = True  # this is hardcoded to the architecture
HM_PARSER_PARAMS = {"max_num_people": 30,
                    "detection_threshold": 0.1,
                    "tag_threshold": 1.0,
                    "use_detection_val": True,
                    "ignore_too_much": False,
                    "tag_per_joint": False,
                    "nms_ksize": 5,
                    "nms_padding": 2}
PARAM_INIT_FN = torch.nn.init.kaiming_normal_

# preprocessing/augmentation
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STDDEV = [0.229, 0.224, 0.225]
HORIZONTAL_FLIP_P = 0.5
ROTATION_MAX_DEGREES = 45
TRANSLATION_MAX_RATIO = [0.1, 0.1]
SCALE_RANGE = [0.7, 1.3]

# logging
MINIVAL_FILE = "assets/coco_minival2017_100.txt"
TIMESTAMP = make_timestamp(with_tz_output=False)
TXT_LOGPATH = os.path.join("log", "[{}]_{}.log".format(__file__, TIMESTAMP))
TB_LOGDIR = os.path.join("tb_log", "train", "[{}]_{}".format(__file__,
                                                             TIMESTAMP))
TB_ATT_VALDIR = os.path.join("tb_log", "att_val", "[{}]_{}".format(__file__,
                                                                   TIMESTAMP))
# minival
TB_DIAGNOSE_EVERY_BATCHES = 100
MINIVAL_EVERY_BATCHES = 150000
SNAPSHOT_DIR = os.path.join("models", "snapshots")
MINIVAL_GT_STDDEVS = [2.0]

# #############################################################################
# # MAIN ROUTINE
# #############################################################################
cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True


STUD_CLASS, INPLANES = AttentionStudentSteps, 80
# LOAD_TIMESTAMP, LOAD_EPOCH, LOAD_STEP = "18_May_2020_14:45:20.437", 13, 3151
# LOAD_TIMESTAMP, LOAD_EPOCH, LOAD_STEP = "20_May_2020_02:23:10.079", 69, 18901
LOAD_TIMESTAMP, LOAD_EPOCH, LOAD_STEP = "21_May_2020_03:46:23.329", 114, 19436
student = STUD_CLASS(MODEL_PATH,
                     DEVICE,
                     INPLANES,
                     NUM_HEATMAPS, 0,  # AE_DIMENSIONS,
                     HALF_PRECISION,
                     PARAM_INIT_FN,
                     False,  # TRAINABLE_STEM
)

inpath = os.path.join(SNAPSHOT_DIR, "{}_epoch{}_step{}".format(
    LOAD_TIMESTAMP, LOAD_EPOCH, LOAD_STEP))
student.load_state_dicts(inpath)

# LOSS FN
att_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(1) * 7).to(DEVICE)

# INSTANTIATE DATALOADERS
with open(MINIVAL_FILE, "r") as f:
    MINIVAL_IDS = [int(line.rstrip('.jpg\n')) for line in f]

# with open(EASY_VAL_SMALL_PATH, "r") as f:
#     EASY_IDS = [int(line.rstrip('.jpg\n')) for line in f]


IMG_NORMALIZE_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=IMG_NORM_MEAN,
                                     std=IMG_NORM_STDDEV,
                                     inplace=True)])

val_dl = torch.utils.data.DataLoader(
    CocoDistillationDatasetAugmented2(COCO_DIR, "val2017",
                                      img_transform=IMG_NORMALIZE_TRANSFORM,
                                      remove_images_without_annotations=False,
                                      # whitelist_ids=MINIVAL_IDS,
                                      gt_stddevs_pix=MINIVAL_GT_STDDEVS),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True)


# MAIN LOOP
txt_logger = ColorLogger(__file__, TXT_LOGPATH, filemode="w")  # w = write anew
tb_logger = SummaryWriter(log_dir=TB_ATT_VALDIR)
student.eval()
with torch.no_grad():
    for i, (img_id, imgs, masks, hms, _, _, segmsks,
            imgs_alt) in enumerate(val_dl, 1):
        txt_logger.info("VALIDATION img: {}".format(i))
        #
        imgs = imgs.to(DEVICE)
        imgs_alt = imgs_alt.to(DEVICE)
        # Potentially you can run into this issue:
        # https://discuss.pytorch.org/t/out-of-memory-error-during-evaluation-but-training-works-fine/12274/7
        # Suggested fix didn't work here... what worked is to trigger pdb
        # right before the out-of-memory line, and then continuing. Somehow
        # the GPU memory gets collected that way. Calling torch.cuda.empty_cache
        # didn't work either
        att, _ = student(imgs, alt=imgs_alt)
        #
        segmsks = torch.nn.functional.interpolate(
            segmsks.unsqueeze(1), att.shape[-2:], mode="bilinear").to(DEVICE)
        att_loss = att_loss_fn(att, segmsks)
        #
        txt_logger.info("att loss: {}".format(att_loss))
        tb_logger.add_scalar("validation att loss", att_loss, i)
        tb_logger.add_scalar("validation img id", img_id, i)
        # finally plot results
        if PLOT_EVERY is not None and i % PLOT_EVERY == 0:
            matplotlib.use("TkAgg")
            #
            imgs -= imgs.min()
            imgs /= imgs.max()
            plot_arrays(imgs[0].permute(1, 2, 0).cpu(), att[0, 0].cpu(),
                        share_zoom=False)
            # tb_logger.add_image("batch imgs", norm_img,
            #                     global_step=global_step)
