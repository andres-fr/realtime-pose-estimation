# -*- coding:utf-8 -*-


"""
TODO:

I was passing a post-sigmoid to BCEWithLogits, and still worked. This is bad?
"""

import os
import time
import math
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
from rtpe.dataloaders import CocoDistillationDatasetAugmented2
from rtpe.students import AttentionStudentSteps
from rtpe.optimization import get_sgd_optimizer, SgdrScheduler, \
    DistillationBceLossKeypointMining
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
ROTATION_MAX_DEGREES = 45
TRANSLATION_MAX_RATIO = [0.1, 0.1]
SCALE_RANGE = [0.7, 1.3]

# training
TRAINABLE_STEM = False
TRAIN_BATCH_SIZE = 16
NUM_EPOCHS = 20000
BATCHNORM_MOMENTUM = 0.1
TRAIN_HW = [450, 450]
MINIVAL_GT_STDDEVS = [2.0]
VAL_GT_STDDEVS = [2.0]
TRAIN_GT_STDDEVS = [7.0]  # [20.0, 5.0]
DISTILLATION_ALPHA = 0.8  # 0.5
OPT_INIT_PARAMS = {"momentum": 0.9, "weight_decay": 0.0003}
SCHEDULER_HYPERPARS = {"max_lr": 0.025,
                       "min_lr": 0.003,
                       "period": 700,
                       "scale_max_lr": 1.02,
                       "scale_min_lr": 1.0,
                       "scale_period": 1.01}

# logging
MINIVAL_FILE = "assets/coco_minival2017_100.txt"
TIMESTAMP = make_timestamp(with_tz_output=False)
TXT_LOGPATH = os.path.join("log", "[{}]_{}.log".format(__file__, TIMESTAMP))
TB_LOGDIR = os.path.join("tb_log", "train", "[{}]_{}".format(__file__,
                                                             TIMESTAMP))

# minival
TB_DIAGNOSE_EVERY_BATCHES = 500
MINIVAL_EVERY_BATCHES = 150000
SNAPSHOT_DIR = os.path.join("models", "snapshots")


class DecayingDivisor:
    """
    At step=0 (when called first), it returns initial_val. When further
    called, it returns a value that exponentially decays to 1.
    """
    def __init__(self, initial_val=20, step_decay=0.003):
        """
        """
        self.initial_val = initial_val
        self._x0 = initial_val - 1
        self.step_decay = step_decay
        self._step = 0

    def __call__(self):
        """
        """
        val = 1 + self.initial_val * math.exp(-self.step_decay * self._step)
        self._step += 1
        return val

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
DUMMY_INPUT = torch.rand(1, 3, 456, 456).to(DEVICE)

STUD_CLASS, INPLANES = AttentionStudentSteps, 80
DECAYING_DIVISOR = DecayingDivisor(20, 0.001)

student = STUD_CLASS(MODEL_PATH,
                     DEVICE,
                     INPLANES,
                     NUM_HEATMAPS, 0,  #  AE_DIMENSIONS,
                     HALF_PRECISION,
                     PARAM_INIT_FN,
                     TRAINABLE_STEM,
                     BATCHNORM_MOMENTUM)

# load pretrained attention part



# LOAD_TIMESTAMP, LOAD_EPOCH, LOAD_STEP = "21_May_2020_00:02:57.265", 3, 1349
# inpath = os.path.join(SNAPSHOT_DIR, "{}_epoch{}_step{}".format(
#     LOAD_TIMESTAMP, LOAD_EPOCH, LOAD_STEP))
# student.load_state_dicts(inpath)


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
DET_POS_WEIGHT = 100  # 100 means that black happens 100 more times than white
det_loss_fn = DistillationBceLossKeypointMining(DET_POS_WEIGHT, DET_POS_WEIGHT, DEVICE)
# att_loss_fn = torch.nn.BCELoss(pos_weight=torch.ones(1)*7).to(DEVICE) THIS SHOULD BE THE LOSS TO USE BUT DOESNT HAVE POS_WEIGHT AND THE OTHER WORKS AMD THE GPU IS BLOATED, SO WE KEEP WITH LOGITS ATM ALTHOUGH WE PROVIDE SIGMOID.
att_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(1)*7).to(DEVICE)
# If stem is not trainable it already has torch.no_grad so opt won't train it
params = (# list(student.mid_stem.parameters()) +
          list(student.att_lo.parameters()) +
          list(student.att_mid.parameters()) +
          list(student.att_hi.parameters()) +
          list(student.att_top.parameters()))
att_opt = get_sgd_optimizer(params, half_precision=HALF_PRECISION,
                            **OPT_INIT_PARAMS)
att_lr_scheduler = SgdrScheduler(att_opt.optimizer, **SCHEDULER_HYPERPARS)
params = (list(student.mid_stem.parameters()) +
          list(student.steps.parameters()) +
          list(student.alt_img_stem.parameters()))
det_opt = get_sgd_optimizer(params, half_precision=HALF_PRECISION,
                            **OPT_INIT_PARAMS)
det_lr_scheduler = SgdrScheduler(det_opt.optimizer, **SCHEDULER_HYPERPARS)


# INSTANTIATE DATALOADERS
with open(MINIVAL_FILE, "r") as f:
    MINIVAL_IDS = [int(line.rstrip('.jpg\n')) for line in f]

with open(EASY_VAL_SMALL_PATH, "r") as f:
    EASY_IDS = [int(line.rstrip('.jpg\n')) for line in f]


IMG_NORMALIZE_TRANSFORM = torchvision.transforms.Compose([
    # jitter? to gray?
    torchvision.transforms.Normalize(mean=IMG_NORM_MEAN,
                                     std=IMG_NORM_STDDEV,
                                     inplace=True)])

AUGMENTATION_TRANSFORM = SeededCompose([
    torchvision.transforms.ToPILImage(mode="F"),
    torchvision.transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_P),
    torchvision.transforms.RandomAffine(
        degrees=(-ROTATION_MAX_DEGREES, +ROTATION_MAX_DEGREES),
        translate=TRANSLATION_MAX_RATIO, scale=SCALE_RANGE),
    torchvision.transforms.RandomCrop(size=TRAIN_HW, pad_if_needed=True),
    torchvision.transforms.ToTensor()])

minival_dl = torch.utils.data.DataLoader(
    CocoDistillationDatasetAugmented2(COCO_DIR, "val2017",
                                     img_transform=IMG_NORMALIZE_TRANSFORM,
                                     remove_images_without_annotations=False,
                                     gt_stddevs_pix=MINIVAL_GT_STDDEVS,
                                     whitelist_ids=MINIVAL_IDS),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True)

val_dl = torch.utils.data.DataLoader(
    CocoDistillationDatasetAugmented2(COCO_DIR, "val2017",
                                     img_transform=IMG_NORMALIZE_TRANSFORM,
                                     remove_images_without_annotations=False,
                                     gt_stddevs_pix=VAL_GT_STDDEVS),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True)


train_ds = CocoDistillationDatasetAugmented2(
    COCO_DIR, "val2017",
    HRNET_VAL_DIR,
    gt_stddevs_pix=TRAIN_GT_STDDEVS,
    img_transform=IMG_NORMALIZE_TRANSFORM,
    overall_transform=AUGMENTATION_TRANSFORM,
    # whitelist_ids=EASY_IDS,
    # whitelist_ids=MINIVAL_IDS,
    remove_images_without_annotations=True,
    )



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
    for (img_id, imgs, masks, hms, teach_hms, teach_aes, segmsks,
         imgs_alt) in train_dl:
        #
        txt_logger.info("TRAINING epoch: {}, global step: {}".format(
            epoch, global_step))

        student.train()
        att_opt_lr = att_opt.optimizer.param_groups[0]["lr"]
        att_opt.zero_grad()
        det_opt_lr = det_opt.optimizer.param_groups[0]["lr"]
        det_opt.zero_grad()
        #
        with torch.no_grad():
            imgs = imgs.to(DEVICE)
            imgs_alt = imgs_alt.to(DEVICE)
        # gt = [torch.cat([gg, teach_aes[:, 0:1, :, :]], dim=1).to(DEVICE)
        #       for gg in hms]
        # masks = masks.unsqueeze(1).expand(*gt[0].shape).to(DEVICE)

        # masks = masks.unsqueeze(1).expand(*gt[0].shape).to(DEVICE)
        # teach_preds = torch.cat([teach_hms, teach_aes[:, 0:1, :, :]],
        #                         dim=1).to(DEVICE)

        att, det = student(imgs, out_hw=TRAIN_HW, alt=imgs_alt,
                           att_divisor=DECAYING_DIVISOR())
        with torch.no_grad():
            segmsks = torch.nn.functional.interpolate(
                segmsks.unsqueeze(1), att.shape[-2:], mode="bilinear").to(DEVICE)
            gt_hms = torch.nn.functional.interpolate(
                hms[0], det.shape[-2:], mode="bilinear").to(DEVICE)
            teacher_hms = torch.nn.functional.interpolate(
                teach_hms, det.shape[-2:], mode="bilinear").to(DEVICE)
            masks = torch.nn.functional.interpolate(
                masks.unsqueeze(1), det.shape[-2:],
                mode="bilinear").expand(gt_hms.shape).to(DEVICE)
        # train segmentation
        segmentation_loss = att_loss_fn(att, segmsks)
        segmentation_loss.backward(retain_graph=True)
        att_opt.step()
        att_lr_scheduler.step()
        # train keypoints
        detection_loss = det_loss_fn(det, teacher_hms, gt_hms,
                                     alpha=DISTILLATION_ALPHA, mask=masks,
                                     background_factor=1)
        detection_loss.backward()
        det_opt.step()
        det_lr_scheduler.step()
        #
        tb_logger.add_scalar("attention loss", segmentation_loss, global_step)
        tb_logger.add_scalar("keypoints loss", detection_loss, global_step)
        tb_logger.add_scalar("attention lrate", att_opt_lr, global_step)
        tb_logger.add_scalar("keypoints lrate", det_opt_lr, global_step)
        txt_logger.info("   attention_batch_loss: {}, lrate: {}".format(
            segmentation_loss, att_opt_lr))
        txt_logger.info("   keypoints_batch_loss: {}, lrate: {}".format(
            detection_loss, det_opt_lr))
        if global_step % TB_DIAGNOSE_EVERY_BATCHES == 0:
            # add image to TB
            norm_img = imgs[0] - imgs[0].min()
            norm_img /= norm_img.max()
            tb_logger.add_image("batch imgs", norm_img,
                                global_step=global_step)
            # add masks
            tb_logger.add_image("gradient masks",
                                masks[0].max(dim=0)[0].unsqueeze(0),
                                global_step=global_step)
            tb_logger.add_image("attention masks", segmsks[0, 0:1],
                                global_step=global_step)
            # add ground truths to TB
            for jj, hm in enumerate(hms, 1):
                tb_logger.add_image("GT heatmaps_{}".format(jj),
                                    hm[0].max(dim=0)[0].unsqueeze(0),
                                    global_step=global_step)
            # add predictions to TB
            tb_logger.add_image("attention maps", att[0],
                                global_step=global_step)

            tb_preds = []
            for jj, d in enumerate(det[0], 1):
                with torch.no_grad():
                    # d = d.clamp(0, 1)
                    d = d.sigmoid()
                    # d = d - d.min()
                    # d /= d.max()
                tb_preds.append(d.unsqueeze(0))
            tb_logger.add_images("detection maps", tb_preds, dataformats="CHW",
                                 global_step=global_step)

            #     tb_logger.add_images("pred_stage_{}".format(jj), tb_preds,
            #                          global_step=global_step,
            #                          dataformats="CHW")
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
    # after every epoch...

    outpath = os.path.join(SNAPSHOT_DIR, "{}_epoch{}_step{}".format(
        TIMESTAMP, epoch, global_step))
    torch.save(student.mid_stem.state_dict(), outpath + "mid_stem.statedict")
    torch.save(student.att_lo.state_dict(), outpath + "att_lo.statedict")
    torch.save(student.att_mid.state_dict(), outpath + "att_mid.statedict")
    torch.save(student.att_hi.state_dict(), outpath + "att_hi.statedict")
    torch.save(student.att_top.state_dict(), outpath + "att_top.statedict")
    torch.save(student.steps.state_dict(), outpath + "steps.statedict")
    torch.save(student.alt_img_stem.state_dict(), outpath + "alt_img_stem.statedict")
    txt_logger.info("Saved snapshot to {}".format(outpath))

txt_logger.info("PROGRAM FINISHED")
