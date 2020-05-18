# -*- coding:utf-8 -*-


"""
TODO:

we need a student that is ABLE to output heatmaps, so:

1. Change the stem for a custom one
2. Train on the student output resolution, i.e. scale down labels not opposite
3. reduce background gradients even more?
4. increase GT heatmap size

"""


import os
import time
#
import torch
import torch.backends.cudnn as cudnn
import torchvision
from tensorboardX import SummaryWriter
#
from rtpe.third_party.group import HeatmapParser
#
from rtpe.helpers import SeededCompose, make_timestamp, ColorLogger, \
    ModuleSummary
from rtpe.dataloaders import CocoDistillationDatasetAugmented
from rtpe.students import CamStudent
from rtpe.optimization import get_sgd_optimizer, SgdrScheduler, \
    DistillationLossKeypointMining
from rtpe.engine import eval_student


# #############################################################################
# # GLOBALS
# #############################################################################
# DEBUG_REDUCE_TRAINSET = True  # remove this once training succeeds
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
ROTATION_MAX_DEGREES = 30
TRANSLATION_MAX_RATIO = [0.1, 0.1]
SCALE_RANGE = [0.7, 1.3]

# training
TRAINABLE_STEM = False
TRAIN_BATCH_SIZE = 10
NUM_EPOCHS = 10000
BATCHNORM_MOMENTUM = 0.1
TRAIN_HW = [450, 450]
MINIVAL_GT_STDDEVS = [2.0]
VAL_GT_STDDEVS = [2.0]
TRAIN_GT_STDDEVS = [20.0]  # [20.0, 5.0]
DISTILLATION_ALPHA = 0.5
OPT_INIT_PARAMS = {"momentum": 0.9, "weight_decay": 0.001}
SCHEDULER_HYPERPARS = {"max_lr": 0.01,
                       "min_lr": 0.001,
                       "period": 500,
                       "scale_max_lr": 1.0,
                       "scale_min_lr": 1.0,
                       "scale_period": 1.0}

# logging
MINIVAL_FILE = "assets/coco_minival2017_100.txt"
TIMESTAMP = make_timestamp(with_tz_output=False)
TXT_LOGPATH = os.path.join("log", "[{}]_{}.log".format(__file__, TIMESTAMP))
TB_LOGDIR = os.path.join("tb_log", "train", "[{}]_{}".format(__file__,
                                                             TIMESTAMP))

# minival
TB_DIAGNOSE_EVERY_BATCHES = 20
MINIVAL_EVERY_BATCHES = 150000
SNAPSHOT_DIR = os.path.join("models", "snapshots")


# #############################################################################
# # MAIN ROUTINE
# #############################################################################
cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

# INSTANTIATE LOGGER
txt_logger = ColorLogger(__file__, TXT_LOGPATH, filemode="w")  # w = write anew
tb_logger = SummaryWriter(log_dir=TB_LOGDIR)

# INSTANTIATE MODEL
DUMMY_INPUT = torch.rand(1, 3, 200, 200).to(DEVICE)

student = CamStudent(None,  # MODEL_PATH,
                     DEVICE,
                     48, 3,  # inplanes, num_stages
                     # 80, 3,  # inplanes, num_stages
                     NUM_HEATMAPS, 0,  #  AE_DIMENSIONS,
                     HALF_PRECISION,
                     PARAM_INIT_FN,
                     True,  # TRAINABLE_STEM,
                     BATCHNORM_MOMENTUM)

hm_parser = HeatmapParser(num_joints=NUM_HEATMAPS,
                          **HM_PARSER_PARAMS)

# LOG MODEL AND HYPERPARAMETERS
student_summary = ModuleSummary.get_model_summary(student, as_string=True)
txt_logger.info(student_summary)
tb_logger.add_text("Architecture summary", student_summary, 0)
# tb_logger.add_graph(student, DUMMY_INPUT)

HPARS_DICT = {"num heatmaps": NUM_HEATMAPS,  # model arch
              "AE dimensions": AE_DIMENSIONS,
              "pretrained HRNet path": MODEL_PATH,
              "half precision": HALF_PRECISION,
              "param init fn": PARAM_INIT_FN,
              **HM_PARSER_PARAMS,
              # preprocessing/augmentation
              "img norm mean": IMG_NORM_MEAN,
              "img norm stddev": IMG_NORM_STDDEV,
              "horizontal flip prob": HORIZONTAL_FLIP_P,
              "rotation max deg": ROTATION_MAX_DEGREES,
              "translation max ratio": TRANSLATION_MAX_RATIO,
              "scale range": SCALE_RANGE,
              # training
              "num epochs": NUM_EPOCHS,
              "trainable stem": TRAINABLE_STEM,
              "train batch size": TRAIN_BATCH_SIZE,
              "batchnorm momentum": BATCHNORM_MOMENTUM,
              **OPT_INIT_PARAMS,
              "train HW": TRAIN_HW,
              "minival_gt_stddevs": MINIVAL_GT_STDDEVS,
              "val_gt_stddevs": VAL_GT_STDDEVS,
              "train_gt_stddevs": TRAIN_GT_STDDEVS,
              "distillation_alpha": DISTILLATION_ALPHA,
              **SCHEDULER_HYPERPARS}
HPARS_DICT = {str(k): str(v) for k, v in HPARS_DICT.items()}
tb_logger.add_hparams(HPARS_DICT, {})
txt_logger.info("HYPERPARAMETERS:\n{}".format(HPARS_DICT))


# INSTANTIATE OPTIMIZER
loss_fn = DistillationLossKeypointMining()
# If stem is not trainable it already has torch.no_grad so opt won't train it
opt = get_sgd_optimizer(student.parameters(), half_precision=HALF_PRECISION,
                        **OPT_INIT_PARAMS)
lr_scheduler = SgdrScheduler(opt.optimizer, **SCHEDULER_HYPERPARS)


# INSTANTIATE DATALOADERS
with open(MINIVAL_FILE, "r") as f:
    MINIVAL_IDS = [int(line.rstrip('.jpg\n')) for line in f]

with open(EASY_VAL_MED_PATH, "r") as f:
    EASY_IDS = [int(line.rstrip('.jpg\n')) for line in f]


IMG_NORMALIZE_TRANSFORM = torchvision.transforms.Compose([
    # jitter? to gray?
    torchvision.transforms.Normalize(mean=IMG_NORM_MEAN,
                                     std=IMG_NORM_STDDEV,
                                     inplace=True)])

AUGMENTATION_TRANSFORM = SeededCompose([
    torchvision.transforms.ToPILImage(mode="F"),
    # torchvision.transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_P),
    # torchvision.transforms.RandomAffine(
    #     degrees=(-ROTATION_MAX_DEGREES, +ROTATION_MAX_DEGREES),
    #     translate=TRANSLATION_MAX_RATIO, scale=SCALE_RANGE),
    torchvision.transforms.RandomCrop(size=TRAIN_HW, pad_if_needed=True),
    torchvision.transforms.ToTensor()])

minival_dl = torch.utils.data.DataLoader(
    CocoDistillationDatasetAugmented(COCO_DIR, "val2017",
                                     img_transform=IMG_NORMALIZE_TRANSFORM,
                                     remove_images_without_annotations=False,
                                     gt_stddevs_pix=MINIVAL_GT_STDDEVS,
                                     whitelist_ids=MINIVAL_IDS),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True)

val_dl = torch.utils.data.DataLoader(
    CocoDistillationDatasetAugmented(COCO_DIR, "val2017",
                                     img_transform=IMG_NORMALIZE_TRANSFORM,
                                     remove_images_without_annotations=False,
                                     gt_stddevs_pix=VAL_GT_STDDEVS),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True)


train_ds = CocoDistillationDatasetAugmented(
    COCO_DIR, "val2017",
    HRNET_VAL_DIR,
    gt_stddevs_pix=TRAIN_GT_STDDEVS,
    img_transform=IMG_NORMALIZE_TRANSFORM,
    overall_transform=AUGMENTATION_TRANSFORM,
    whitelist_ids=EASY_IDS,
    remove_images_without_annotations=True)


# train_ds = CocoDistillationDatasetAugmented(
#     COCO_DIR, "train2017",
#     HRNET_TRAIN_DIR,
#     gt_stddevs_pix=TRAIN_GT_STDDEVS,
#     img_transform=IMG_NORMALIZE_TRANSFORM,
#     overall_transform=AUGMENTATION_TRANSFORM,
#     remove_images_without_annotations=True)
# # THIS IS TO TEST IF THE NN LEARNS AT ALL! REMOVE IT TO PROPERLY TRAIN
# if DEBUG_REDUCE_TRAINSET:
#     train_ds.ids = train_ds.ids[:50]


train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True)


# MAIN LOOP
global_step = 1
best_score = -1
for epoch in range(NUM_EPOCHS):
    for (img_id, imgs, masks, hms, teach_hms, teach_aes) in train_dl:
        txt_logger.info("TRAINING epoch: {}, global step: {}".format(
            epoch, global_step))
        student.train()
        opt_lr = opt.optimizer.param_groups[0]["lr"]
        opt.zero_grad()
        #
        imgs = imgs.to(DEVICE)
        gt = [torch.cat([gg, teach_aes[:, 0:1, :, :]], dim=1).to(DEVICE)
              for gg in hms]
        masks = masks.unsqueeze(1).expand(*gt[0].shape).to(DEVICE)
        teach_preds = torch.cat([teach_hms, teach_aes[:, 0:1, :, :]],
                                dim=1).to(DEVICE)
        # this particular model has the option of intermediate learning,
        # so the loss needs to be adapted in that case
        preds = student(imgs, out_hw=TRAIN_HW)
        # batch_loss = sum([loss_fn(pp, teach_preds, gg,
        #                           alpha=DISTILLATION_ALPHA, mask=masks)
        #                   for pp, gg in zip(preds, gt)])
        batch_loss = loss_fn(preds[0][:, :17], teach_preds[:, :17],
                             gt[0][:, :17],
                             alpha=DISTILLATION_ALPHA, mask=masks[:, :17],
                             background_factor=0.01)  ###
        #
        batch_loss.backward()
        opt.step()
        lr_scheduler.step()
        #
        tb_logger.add_scalar("batch_loss", batch_loss, global_step)
        tb_logger.add_scalar("lrate", opt_lr, global_step)
        txt_logger.info("   batch_loss: {}, lrate: {}".format(batch_loss,
                                                              opt_lr))
        if global_step % TB_DIAGNOSE_EVERY_BATCHES == 0:
            # add image to TB
            norm_img = imgs[0] - imgs[0].min()
            norm_img /= norm_img.max()
            tb_logger.add_image("batch imgs", norm_img,
                                global_step=global_step)
            # add ground truths to TB
            for jj, hm in enumerate(hms, 1):
                tb_logger.add_image("GT heatmaps_{}".format(jj),
                                    hm[0].max(dim=0)[0].unsqueeze(0),
                                    global_step=global_step)
            # add predictions to TB
            tb_preds = []
            for jj, pp in enumerate(preds, 1):
                for p in pp[0]:
                    tbp = p - p.min()
                    tbp /= tbp.max()
                    tb_preds.append(tbp.unsqueeze(0))
                tb_logger.add_images("pred_stage_{}".format(jj), tb_preds,
                                     global_step=global_step,
                                     dataformats="CHW")
            # add weights and gradients histogram
            for name, param in student.named_parameters():
                tb_logger.add_histogram(name + "_PARAMETERS",
                                        param.cpu().data.numpy(),
                                        global_step)
                if param.grad is not None:
                    tb_logger.add_histogram(name + "_GRADIENTS",
                                            param.grad.cpu().data.numpy(),
                                            global_step)
        # #
        # if global_step % MINIVAL_EVERY_BATCHES == 0:
        #     with torch.no_grad():
        #         t = time.time()
        #         minval_evaluation = eval_student(student, hm_parser,
        #                                          minival_dl,
        #                                          DEVICE,
        #                                          plot_every=None,
        #                                          save_every=None,
        #                                          save_dir="/tmp")
        #         minval_evaluation = dict(minval_evaluation)

        #         elapsed = time.time() - t
        #         txt_logger.info("MINIVAL METRICS: {}".format(
        #             minval_evaluation))
        #         txt_logger.info("minival elapsed seconds: {}".format(elapsed))
        #         tb_logger.add_text("minival metrics", str(minval_evaluation),
        #                            global_step)
        #         tb_logger.add_scalar("minival elapsed_seconds", elapsed,
        #                              global_step)
        #         if minval_evaluation["AP"] > best_score:
        #             best_score = minval_evaluation["AP"]
        #             # snapshot_path = os.path.join(SNAPSHOT_DIR,
        #             #                              make_timestamp() + ".statedict")
        #             outpath = os.path.join(SNAPSHOT_DIR, TIMESTAMP +
        #                                    ".statedict")
        #             student.save_body(outpath)
        #             txt_logger.info("Saved snapshot to {}".format(outpath))
        # #
        global_step += 1
    # # after every epoch...
    # outpath = os.path.join(SNAPSHOT_DIR, "{}_epoch{}_step{}.statedict".format(
    #     TIMESTAMP, epoch, global_step))
    # student.save_body(outpath)
    # txt_logger.info("Saved snapshot to {}".format(outpath))

txt_logger.info("PROGRAM FINISHED")
