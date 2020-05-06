# -*- coding:utf-8 -*-

"""
This module holds functionality to perform distillation from a Higher-HRNet:
https://arxiv.org/abs/1908.10357

Some of the code as well as the pre-trained models were borrowed from their
official repo (thanks a lot!), particularly all the ``rtpe.third_party`` code:

https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation

The distillation loss function is also inspired in Fast Human Pose Estimation
(CVPR 2019), only details differ.


TACTICAL TODO:

* define students, and train them. The training script should be
  clear but flexible to contain all use cases.

  * Idea 1: Progressive Context Refinement: several pipelines of different
    resolutions, where the HM output gets added from coarse to fine

  * Idea 2: Replace ResNet18 with step convolutions

  * Idea 3: Include many-to-many scale fusions like in HRNet.

  * Idea 4: Mix them all
"""

import time
import os
import logging
#
import torch
import torch.backends.cudnn as cudnn
import torchvision
from tensorboardX import SummaryWriter
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
#
from rtpe.third_party.utils import get_model_summary, create_logger
from rtpe.third_party.fp16_utils.fp16util import network_to_half
from rtpe.third_party.group import HeatmapParser
from rtpe.third_party.vis import save_valid_image
#
from rtpe.helpers import SeededCompose, plot_arrays
from rtpe.students import RefinerStudent
from rtpe.optimization import MaskedMseLoss, get_sgd_optimizer, \
    SgdrScheduler, DistillationLoss
from rtpe.dataloaders import CocoDistillationDatasetAugmented



# #############################################################################
# # MAIN ROUTINE
# #############################################################################

cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True


HOME = os.path.expanduser("~")
VAL_DIR = os.path.join(HOME, "datasets/coco/val2017")
MODEL_PATH = "models/pose_higher_hrnet_w48_640.pth.tar"
HALF_PRECISION = True  # this is hardcoded to the architecture
DEVICE = "cuda"
BATCH_SIZE = 4
NUM_EPOCHS = 5

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

minival_dataset = CocoDistillationDatasetAugmented(
    "/home/a9fb1e/datasets/coco", "val2017",
    "/home/a9fb1e/datasets/coco/hrnet_predictions/val2017",
    # img_transform=IMG_NORMALIZE_TRANSFORM,
    # overall_transform=OVERALL_HHRNET_TRANSFORM,
    whitelist_ids=MINIVAL_IDS
)

loss = DistillationLoss()

hm_parser = HeatmapParser(num_joints=17,
                          max_num_people=30,
                          detection_threshold=0.1,
                          tag_threshold=1.0,
                          use_detection_val=True,
                          ignore_too_much=False,
                          tag_per_joint=True,
                          nms_ksize=5,
                          nms_padding=2)

# breakpoint()  # matplotlib.use("TkAgg")
all_preds = []
all_scores = []
for i in range(len(minival_dataset)):
    print(">>>", i)
    img_id, img, mask, hms, teach_hms, teach_ae = minival_dataset[i]  # 3
    dist_loss = loss(teach_hms, hms[0], hms[0], mask=mask)
    grouped, scores = hm_parser.parse(
        teach_hms.unsqueeze(0), teach_ae.unsqueeze(0), adjust=True, refine=True)
    final_results = [x for x in grouped[0] if x.size > 0]
    # # breakpoint()
    # save_valid_image(
    #     img.sub_(img.min()).mul_(255.0 / img.max()).cpu().permute(1, 2, 0).numpy(),
    #     [x for x in grouped[0] if x.size > 0], "test_{}.jpg".format(i),
    #     dataset="COCO")
    # # breakpoint()
    # print(">>>", i, dist_loss, "!!!!!!!!!", [x.size for x in grouped])


    # plot_img = (img - img.min()).permute(1, 2, 0)
    # plot_img /= plot_img.max()
    # plot_arrays(img.permute(1, 2, 0), mask, hms.sum(dim=0), teach_hms.sum(dim=0))
    # breakpoint()


    all_preds.append(final_results)
    all_scores.append(scores)

# name_values, _ = test_dataset.evaluate(
#     cfg, all_preds, all_scores, final_output_dir)

eval_dict, mAP = minival_dataset.evaluate(all_preds, all_scores, ".", False, False)
eval_str = "\n".join([k+"="+str(v) for k, v in eval_dict.items()])
print(eval_str)
breakpoint()













cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True


HOME = os.path.expanduser("~")
VAL_DIR = os.path.join(HOME, "datasets/coco/val2017")
MODEL_PATH = "models/pose_higher_hrnet_w48_640.pth.tar"
INPUT_SIZE = 640  # this is hardcoded to the architecture
VERBOSE = True
HALF_PRECISION = True  # this is hardcoded to the architecture
DEVICE = "cuda"
MINIVAL_SIZE = 500
BATCH_SIZE = 4
NUM_EPOCHS = 5
DUMMY_INPUT = torch.rand((1, 3, INPUT_SIZE, INPUT_SIZE)).to(DEVICE)





# THIS SNIPPET RUNS A TEST OPTIMIZATION
#
s1 = StudentLinear(MODEL_PATH, HALF_PRECISION, trainable_stem=False)
loss_fn = MaskedMseLoss()
opt = get_sgd_optimizer(s1.parameters(), half_precision=HALF_PRECISION)
lr_scheduler = SgdrScheduler(opt.optimizer, max_lr=0.001, min_lr=0.001,
                             period=200,
                             scale_max_lr=1.0, scale_min_lr=1.0,
                             scale_period=1.0)
LOGGER, tb_log_dir = create_logger("test_log", "log", "valid")
TB_LOGGER = SummaryWriter(log_dir=os.path.join("tb_log", tb_log_dir))


dummy_data = torch.rand(100, BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
dummy_targets = torch.rand(100, BATCH_SIZE, 1, 123, 123).to(DEVICE)
dummy_targets[:, :, :, 10:, :] = 0  # some dummy targeting
mask = torch.ones(BATCH_SIZE, 1, 123, 123).to(DEVICE)
mask[:, :, :10, :] = 0
t = time.time()
global_step = 1
for epoch in range(NUM_EPOCHS):
    for b_i, (batch, targets) in enumerate(zip(dummy_data, dummy_targets)):
        print("epoch", epoch, "step", global_step)
        opt_lr = opt.optimizer.param_groups[0]["lr"]
        opt.zero_grad()
        preds = s1(batch, out_hw=(123, 123))
        batch_loss = loss_fn(preds, dummy_targets,
                             mask=mask)
        TB_LOGGER.add_scalar("batch_loss", batch_loss, global_step)
        TB_LOGGER.add_scalar("lrate", opt_lr, global_step)
        #
        batch_loss.backward()
        opt.step()
        lr_scheduler.step()
        global_step += 1
eps = time.time() - t
TB_LOGGER.add_scalar("elapsed_seconds", eps, 0)
print("[ELAPSED SECONDS:]", eps)
TB_LOGGER.close()
breakpoint()


# # THIS EXTRACTS A RANDOM MINIVAL SPLIT
# mv, val_rest = make_rand_minival_split(VAL_DIR, MINIVAL_SIZE, extension=".jpg")
# breakpoint()


# # THIS PART LOADS THE HHRNET AND THE STUDENT1, AND PLOTS/TBOARDS THEIR STATS.
# # WE ALREADY TESTED THAT BOTH STEMS ARE IDENTICAL
# s1 = Student1(MODEL_PATH, DEVICE, half_precision=True).eval()
# hhrnet = get_hrnet_w48_teacher(MODEL_PATH).to(DEVICE)
# # TB_LOGGER.add_graph(s1, DUMMY_INPUT)
# # TB_LOGGER.add_graph(hhrnet, DUMMY_INPUT)

# LOGGER.info("HIGHER HRNET")
# LOGGER.info(get_model_summary(hhrnet, DUMMY_INPUT, verbose=True))

# LOGGER.info("STUDENT 1")
# LOGGER.info(get_model_summary(s1, DUMMY_INPUT, verbose=True))
# with torch.no_grad():
#     aaa = hhrnet(DUMMY_INPUT)
#     bbb = s1(DUMMY_INPUT)
# breakpoint()
# print((aaa == bbb).all())

# # TB_LOGGER.close()
# breakpoint()









# breakpoint()

# # LOAD IMAGE
# # IMG_PATH = "data/000000002685.jpg"
# # IMG_PATH = "data/000000001000.jpg"


# coco_t = "/home/a9fb1e/datasets/coco/train2017"
# pp = [os.path.join(coco_t, p) for p in os.listdir(coco_t)]
# for IMG_PATH in pp:
#     preproc_img = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])])
#     #
#     img = Image.open(IMG_PATH)
#     print("processing", IMG_PATH, img.size)
#     resized_img, center, scale = resize_align_multi_scale(np.array(img),
#                                                           INPUT_SIZE, 1, 1)
#     t = preproc_img(resized_img).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         embeddings, heatmaps = model(t)




# HEATMAPS ARE IN THIS ORDER:
# nose, leye, reye, lear, rear, lshould, rshould, lelbow, relbow, lwrist, rwrist,
# lhip, rhip, lknee, rknee, lankle, rankle


# import matplotlib.pyplot as plt

# # # plot img
# # plt.clf(); plt.imshow(resized_img); plt.show()
# # # plot all heatmaps
# # plt.clf(); plt.imshow(heatmaps.squeeze().cpu().sum(0)); plt.show()

# # plot separate heatmaps on the img
# p = torch.nn.functional.interpolate(heatmaps, size=t[0, 0].shape,
#                                     mode="bilinear").squeeze() * 10

# p += t[0, 0]
# breakpoint()

# for pp in p:
#     plt.clf();  plt.imshow(pp.cpu());  plt.show()
