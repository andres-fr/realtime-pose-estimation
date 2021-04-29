# -*- coding:utf-8 -*-


"""
Version of validate_inference trimmed down to perform inference
to find heads.

ffmpeg -i 0481BL.MXF 0481BL.MXF_%03d.png

for i in /shared/mvn1e/sina/*; do python face_extractor.py -i $i; done
for i in <ITER OVER FOLDERS WITH IMGS>; do python face_extractor.py -i $i; done


.. WARNING:

  This script seems to RAM somewhere (around 10GB after 6 hours).
  Cheapest fix is to have enough RAM and run it in small enough batches.
"""

import surgery
import caffe

import os
#
import argparse
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
    else:
        return arr



def groups_to_heads(groups, p_thresh=0.3):
    """
    """
    people = [person for group in groups for person in group]
    heads = [np.float32([(x, y) for x, y, p in person[:4, :3] if p>=p_thresh])
             for person in people]
    return heads

    # heads = [np.float32([(x, y) for x, y, p in g[0][:4, :3] if p>=p_thresh])
    #          for g in groups]
    # return heads


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
argparser = argparse.ArgumentParser()
# argparser.add_argument("-p", "-pose_net_path", required=True,
#                        help="Path to the HigherHRNet PyTorch model")
# argparser.add_argument("-c1", "-caffe_prototxt", required=True,
#                        help="Path to the Caffe FaceSeg prototxt file")
# argparser.add_argument("-c2", "-caffe_model", required=True,
#                        help="Path to the Caffe FaceSeg model file")
argparser.add_argument("-i", "--imgs_dir", required=True,
                       help="Path to a directory with images to process")
argparser.add_argument("-o", "--out_dir", default=None,
                       help="Path to a directory to save segmentations")
argparser.add_argument("--save_mix", action="store_true",
                       help="If given, export also RGB+mask on top \
                       (useful for visualization but takes more disk space)")
args = argparser.parse_args()

# IMG_DIR = "/shared/mvn1e/mocap_library/mairi_png"
# IMG_DIR = "/shared/mvn1e/mocap_library/face_test_frames"
IMG_DIR = args.imgs_dir
SAVE_DIR = args.out_dir
if SAVE_DIR is None:
    SAVE_DIR = IMG_DIR
SAVE_MIX = args.save_mix

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
BBOX_RADIUS = 70 # (pixels)

# #############################################################################
# # MAIN ROUTINE
# #############################################################################
caffe.set_device(0)
caffe.set_mode_gpu()

# import pdb; pdb.set_trace()

net = caffe.Net("../face_segmentation/data/face_seg_fcn8s_deploy.prototxt",
                "../face_segmentation/data/face_seg_fcn8s.caffemodel",
                caffe.TEST)


# dataloaders
IMG_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=IMG_NORM_MEAN,
                                     std=IMG_NORM_STDDEV,
                                     inplace=True)])

# img_paths = [os.path.join(IMG_DIR, p) for p in os.listdir(IMG_DIR) if p.endswith(".png")]
img_paths = [os.path.join(IMG_DIR, p) for p in os.listdir(IMG_DIR) if p.endswith(".jpg")]




# model
hhrnet = get_hrnet_w48_teacher(MODEL_PATH).to(DEVICE)
hhrnet.eval()

hm_parser = HeatmapParser(num_joints=NUM_HEATMAPS,
                          **HM_PARSER_PARAMS)
# import pdb; pdb.set_trace()
# main loop
all_preds = []
all_scores = []
for ii, imgpath in enumerate(img_paths):
    imgname = os.path.splitext(os.path.basename(imgpath))[0]
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


    # hh = hms[0].to("cpu")
    # plot_arrays(hh[0], hh[1])

    grouped = [g for g in grouped if len(g>0)]
    if not grouped:
        continue
    # print(grouped)
    heads = groups_to_heads(grouped, KP_THRESH)
    head_bboxes = []
    for h in heads:
        min_x, max_x = h[:, 0].min(), h[:, 0].max()
        min_y, max_y = h[:, 1].min(), h[:, 1].max()
        head_bboxes.append((min_x, max_x, min_y, max_y))


    # imgclean = Image.open(imgpath).convert("RGB")
    # pltimg = np.float32(imgclean) / 255.0
    # maxhm = hms[0].to("cpu").numpy().max(axis=0)
    # # pltimg[:, :, 0] += maxhm
    # # pltimg[:, :, 1] += maxhm
    # # pltimg[:, :, 2] += maxhm
    # pltimg /= pltimg.max()
    # plot_arrays(pltimg, maxhm)

    # import pdb; pdb.set_trace()


    # SANDBOX: FINDING THE BBOX RADIUS IS CRITICAL FOR THE PERFORMANCE.
    # OPTIMAL RADIUS FOR MULTISCALE AND MISSING KEYPOINTS SHOULD BE IMPROVED
        
    # for each bbox, obtain center and max radius, and redefine bboxes
    head_centers = [(0.5*(x0 + x1), 0.5*(y0 + y1)) for x0, x1, y0, y1 in head_bboxes]
    # HARDCODED RADIUS:
    head_maxrads = [0.5 * max(x1 - x0, y1 - y0) for x0, x1, y0, y1 in head_bboxes]


    head_maxrads = [max(30, x*3) for x in head_maxrads]
    
    # head_maxrads = [BBOX_RADIUS for x0, x1, y0, y1 in head_bboxes]

    expanded_bboxes = [(hc_x - rad, hc_y - rad, hc_x + rad, hc_y + rad) for (hc_x, hc_y), rad in zip(head_centers, head_maxrads)]
    print(">> head centers, expanded bboxes:", head_centers, expanded_bboxes)

    # import pdb; pdb.set_trace()

    ### END SANDBOX






    

    # expanded_bboxes = [(x0, x1, y0, y1) for x0, y0, x1, y1 in expanded_bboxes]
    expanded_bboxes = [expand_bbox(x0, x1, y0, y1,
                                   expansion_ratio=1, clip_wh=img.size)
                       for x0, y0, x1, y1 in expanded_bboxes]

    # print(">>>>", len(grouped), grouped)
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
        print("    >>>>", ha.shape, ha.dtype)
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
        if PLOT:
            plot_arrays(arr, arr_with_mask)
            # plot_arrays(ha_im, np.array(ha_im)*mask[:, :, None],
            #             np.array(ha_im) * (1-mask[:, :, None]))
        # save:
        face_mask_img = Image.fromarray(face_mask)
        face_mask_img.save(os.path.join(SAVE_DIR, imgname + "_mask.png"))
        #
        if SAVE_MIX:
            arr_with_mask = (arr).astype(np.float32)
            arr_with_mask[:, :, 0] += 60*face_mask
            arr_with_mask[:, :, 1] += 60*face_mask
            arr_with_mask[:, :, 2] += 100*face_mask
            # arr_with_mask = (arr + 50*face_mask[:, :, None]).astype(np.float32)
            arr_with_mask *= 255.0 /arr_with_mask.max()
            arr_with_mask = arr_with_mask.astype(np.uint8)
            arr_with_mask_img = Image.fromarray(arr_with_mask)
            arr_with_mask_img.save(os.path.join(SAVE_DIR,
                                                imgname + "_masked.jpg"))
        print(">> Saved to", SAVE_DIR)
