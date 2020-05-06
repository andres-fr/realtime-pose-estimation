# -*- coding:utf-8 -*-

"""
This module holds functionality to perform distillation from a Higher-HRNet:
https://arxiv.org/abs/1908.10357

Most of the code as well as the pre-trained models were borrowed from their
official repo (thanks a lot!):

https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation

The distillation loss function is also inspired in Fast Human Pose Estimation
(CVPR 2019), but details will differ.
"""

import os
import logging
import torch
import torch.backends.cudnn as cudnn
import torchvision
from PIL import Image
import numpy as np
#
from rtpe.helpers import get_hrnet_w48_teacher
from rtpe.third_party.utils import get_model_summary  # , setup_logger  #, create_logger
from rtpe.third_party.transforms import resize_align_multi_scale


# CORRECTIONS TO CFG:
# * pretrained_layers is used by init_weights only
# * stem_inplanes is never used (hardcoded to 64). integrate into self.cfg? check _make_layer, does some weird business with the inplanes
# MODEL->EXTRA->STAGE belongs to make_stage only.
# FUSE_METHOD: unused?

# CFG = {"OUTPUT_DIR": "output", "LOG_DIR": "log", "DATA_DIR": "", "GPUS": (0,),
#        "WORKERS": 4, "PRINT_FREQ": 100, "AUTO_RESUME": True,
#        "PIN_MEMORY": True, "RANK": 0, "VERBOSE": True, "DIST_BACKEND": "nccl",
#        "MULTIPROCESSING_DISTRIBUTED": True,
#        "FP16": {"ENABLED": True, "STATIC_LOSS_SCALE": 1.0,
#                 "DYNAMIC_LOSS_SCALE": True},
#        "CUDNN": {"BENCHMARK": True, "DETERMINISTIC": False, "ENABLED": True},
#        "MODEL": {"NAME": "pose_higher_hrnet", "INIT_WEIGHTS": True,
#                  "PRETRAINED": "models/pytorch/imagenet/hrnet_w48-8ef0771d.pth",
#                  "NUM_JOINTS": 17, "TAG_PER_JOINT": True,
#                  "EXTRA": {"FINAL_CONV_KERNEL": 1, "PRETRAINED_LAYERS": ["*"],
#                            "STEM_INPLANES": 64,
#                            "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2,
#                                       "BLOCK": "BASIC", "NUM_BLOCKS": [4, 4],
#                                       "NUM_CHANNELS": [48, 96],
#                                       "FUSE_METHOD": "SUM"},
#                            "STAGE3": {"NUM_MODULES": 4, "NUM_BRANCHES": 3,
#                                       "BLOCK": "BASIC",
#                                       "NUM_BLOCKS": [4, 4, 4],
#                                       "NUM_CHANNELS": [48, 96, 192],
#                                       "FUSE_METHOD": "SUM"},
#                            "STAGE4": {"NUM_MODULES": 3, "NUM_BRANCHES": 4,
#                                       "BLOCK": "BASIC",
#                                       "NUM_BLOCKS": [4, 4, 4, 4],
#                                       "NUM_CHANNELS": [48, 96, 192, 384],
#                                       "FUSE_METHOD": "SUM"},
#                            "DECONV": {"NUM_DECONVS": 1, "NUM_CHANNELS": [48],
#                                       "KERNEL_SIZE": [4], "NUM_BASIC_BLOCKS": 4,
#                                       "CAT_OUTPUT": [True]}},
#                  "SYNC_BN": False},
#        "LOSS": {"NUM_STAGES": 2, "WITH_HEATMAPS_LOSS": (True, True),
#                 "HEATMAPS_LOSS_FACTOR": (1.0, 1.0),
#                 "WITH_AE_LOSS": (True, False), "AE_LOSS_TYPE": "exp",
#                 "PUSH_LOSS_FACTOR": (0.001, 0.001),
#                 "PULL_LOSS_FACTOR": (0.001, 0.001)},
#        "DATASET": {"ROOT": "data/coco", "DATASET": "coco_kpt",
#                    "DATASET_TEST": "coco", "NUM_JOINTS": 17,
#                    "MAX_NUM_PEOPLE": 30, "TRAIN": "train2017",
#                    "TEST": "val2017", "DATA_FORMAT": "jpg", "MAX_ROTATION": 30,
#                    "MIN_SCALE": 0.75, "MAX_SCALE": 1.5, "SCALE_TYPE": "short",
#                    "MAX_TRANSLATE": 40, "INPUT_SIZE": 640,
#                    "OUTPUT_SIZE": [160, 320],
#                    "FLIP": 0.5, "SIGMA": 2, "SCALE_AWARE_SIGMA": False,
#                    "BASE_SIZE": 256.0, "BASE_SIGMA": 2.0, "INT_SIGMA": False,
#                    "WITH_CENTER": False},
#        "TRAIN": {"LR_FACTOR": 0.1, "LR_STEP": [200, 260], "LR": 0.001,
#                  "OPTIMIZER": "adam", "MOMENTUM": 0.9, "WD": 0.0001,
#                  "NESTEROV": False, "GAMMA1": 0.99, "GAMMA2": 0.0,
#                  "BEGIN_EPOCH": 0, "END_EPOCH": 300, "RESUME": False,
#                  "CHECKPOINT": "", "IMAGES_PER_GPU": 10, "SHUFFLE": True},
#        "TEST": {"IMAGES_PER_GPU": 1, "FLIP_TEST": True, "ADJUST": True,
#                 "REFINE": True, "SCALE_FACTOR": [1],
#                 "DETECTION_THRESHOLD": 0.1, "TAG_THRESHOLD": 1.0,
#                 "USE_DETECTION_VAL": True, "IGNORE_TOO_MUCH": False,
#                 "MODEL_FILE": MODEL_FILE,
#                 "IGNORE_CENTER": True, "NMS_KERNEL": 5, "NMS_PADDING": 2,
#                 "PROJECT2IMAGE": True, "WITH_HEATMAPS": (True, True),
#                 "WITH_AE": (True, False), "LOG_PROGRESS": False},
#        "DEBUG": {"DEBUG": True, "SAVE_BATCH_IMAGES_GT": False,
#                  "SAVE_BATCH_IMAGES_PRED": False, "SAVE_HEATMAPS_GT": True,
#                  "SAVE_HEATMAPS_PRED": True, "SAVE_TAGMAPS_PRED": True}}


# class Resize(object):
#     """
#     Torchvision transform to resize an image using PIL. Source:
#     https://discuss.pytorch.org/t/no-resize-in-torchvision-transforms/10549/3
#     """

#     def __init__(self, maxsize, interpolation=Image.BILINEAR):
#         """
#         :param maxsize: The expected maximal output size in pixels. I.e. if the
#           image is portrait, this will be the output height. If landscape, the
#           width.
#         """
#         self.maxsize = maxsize
#         self.interpolation = interpolation

#     def __call__(self, img):
#         """
#         """
#         w, h = img.size
#         ratio = float(self.maxsize) / max(w, h)
#         out_size = [round(w * ratio), round(h * ratio)]
#         # assert self.maxsize in out_size, "This should never happen!"

#         print(">>>>", out_size)
#         for i, dimvar in enumerate(out_size):
#             if dimvar != self.maxsize and dimvar % 2 != 0:
#                 out_size[i] += 1
#         print(">>>>", out_size)
#         return img.resize(out_size, resample=self.interpolation)

cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True


MODEL_PATH = "models/pose_higher_hrnet_w48_640.pth.tar"
INPUT_SIZE = 640  # this is hardcoded to the architecture
VERBOSE = True
HALF_PRECISION = True  # this is hardcoded to the architecture
DEVICE = "cuda"

# logger, time_str = setup_logger("/log", 0, "valid")
logger = logging.getLogger()


model = get_hrnet_w48_teacher(MODEL_PATH).to(DEVICE)


# LOG MODEL?
DUMMY_INPUT = TORCH.RAND((1, 3, INPUT_SIZE, INPUT_SIZE))
LOGGER.INFO(GET_MODEL_SUMMARY(MODEL, DUMMY_INPUT, VERBOSE=VERBOSE))

breakpoint()

# LOAD IMAGE
# IMG_PATH = "data/000000002685.jpg"
# IMG_PATH = "data/000000001000.jpg"


coco_t = "/home/a9fb1e/datasets/coco/train2017"
pp = [os.path.join(coco_t, p) for p in os.listdir(coco_t)]
for IMG_PATH in pp:
    preproc_img = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
    #
    img = Image.open(IMG_PATH)
    print("processing", IMG_PATH, img.size)
    resized_img, center, scale = resize_align_multi_scale(np.array(img),
                                                          INPUT_SIZE, 1, 1)
    t = preproc_img(resized_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embeddings, heatmaps = model(t)




# HEATMAPS ARE IN THIS ORDER:
# nose, leye, reye, lear, rear, lshould, rshould, lelbow, relbow, lwrist, rwrist,
# lhip, rhip, lknee, rknee, lankle, rankle




import matplotlib.pyplot as plt

# # plot img
# plt.clf(); plt.imshow(resized_img); plt.show()
# # plot all heatmaps
# plt.clf(); plt.imshow(heatmaps.squeeze().cpu().sum(0)); plt.show()

# plot separate heatmaps on the img
p = torch.nn.functional.interpolate(heatmaps, size=t[0, 0].shape,
                                    mode="bilinear").squeeze() * 10

p += t[0, 0]
breakpoint()

for pp in p:
    plt.clf();  plt.imshow(pp.cpu());  plt.show()
