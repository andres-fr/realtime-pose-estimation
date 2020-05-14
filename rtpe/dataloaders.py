# -*- coding:utf-8 -*-


"""
This module holds the dataloaders and their direct data-related dependencies
(e.g. converting keypoints into heatmaps).
"""


import os
import numpy as np
import torch
import torchvision
# from skimage.color import gray2rgb, rgb2lab, rgb2hsv
#
import pycocotools
#
from .third_party.COCODataset import CocoDataset


# #############################################################################
# # GLOBALS
# #############################################################################


# #############################################################################
# # DATA LOADERS
# #############################################################################
class HWHeatmapGenerator:
    """
    This class is used by the COCO datasets defined below to create per-joint
    heatmaps, where each keypoint is converted into a gaussian bell with a
    given standard deviation in pixels.
    It is based on the ``HeatmapGenerator`` class from the third party section,
    but doesn't expect a hardcoded square image shape at construction time.
    """
    def __init__(self, num_joints=17, stddev_pixels=2.0):
        """
        """
        self.num_joints = num_joints
        #
        assert stddev_pixels > 0, "stddev_pixels must be positive"
        self.sigma = stddev_pixels
        size = 6 * self.sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * self.sigma + 1, 3 * self.sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) /
                        (2 * self.sigma ** 2))

    def __call__(self, joints, out_shape_hw):
        """
        See the COCO datasets for usage examples.
        """
        sigma = self.sigma
        out_h, out_w = out_shape_hw
        hms = np.zeros((self.num_joints, out_h, out_w), dtype=np.float32)
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if (x < 0 or y < 0 or x >= out_w or y >= out_h):
                        continue
                    # upper_left, bottom_righ: extract box for heatmap addition
                    ul = (int(np.round(x - 3 * sigma - 1)),
                          int(np.round(y - 3 * sigma - 1)))
                    br = (int(np.round(x + 3 * sigma + 2)),
                          int(np.round(y + 3 * sigma + 2)))
                    c, d = (max(0, -ul[0]),
                            min(br[0], out_w) - ul[0])
                    a, b = (max(0, -ul[1]),
                            min(br[1], out_h) - ul[1])

                    cc, dd = max(0, ul[0]), min(br[0], out_w)
                    aa, bb = max(0, ul[1]), min(br[1], out_h)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms


class CocoDistillationDataset(CocoDataset):
    """
    """
    def __init__(self,
                 coco_root_path,
                 coco_dataset_name,
                 teacher_output_dir=None,
                 remove_images_without_annotations=False,
                 gt_stddevs_pix=[2.0],
                 num_joints=17,
                 whitelist_ids=None):
        """
        For a given annotated COCO image, this dataset returns
        ``(img_id, img, mask, heatmaps, teacher_hms, teacher_ae)``,
        where ``img_id`` is the number as ``LongTensor``, ``img`` is
        a ``FloatTensor(3, h, w)`` with values between 0.0 and 1.0, ``mask``
        is a ``FloatTensor(h, w)`` where the 'difficult' people in the images
        are set to 0.0 (rest is 1.0), ``heatmaps`` is a list of
        ``FloatTensor(num_joints, h, w)`` with the labeled keypoints as
        gaussian blobs (each elt of the list will have a different standard
        deviation), and the ``teacher`` outputs are the respective heatmap
        and associative embedding outputs that have been precomputed previously
        with Higher HRNet (see the ``teacher_inference.py`` script for more
        details on that).

        :param teacher_output_dir: If None, the output ``teacher_hms,
          teacher_ae`` will be dimensionless tensors. If a path is given, the
          method ``_get_teacher_data`` will try to use it to load the
          precomputed teacher predictions
        :param remove_images_without_annots: This keeps all images that have
          at least one non-crowd annotated person. It should NOT be used at
          validation time
        :param gt_stddevs_pix: List with tandard deviations for the ground
          truth heatmap blobs, in pixels. Blobs are gaussian. For each stddev,
          a HWHeatmapGenerator creates a whole set of heatmaps. E.g. if
          ``[4, 2]`` is given, the first heatmaps will have all a stddev of 4
          pixels, and the second set 2. This can be useful for refinement.
        :param whitelist_ids: If given, only images with that ID will be
          included (e.g. ``000000000123.jpg`` will have a ``123`` ID).
        """
        super().__init__(coco_root_path, coco_dataset_name, None)
        self.num_joints = num_joints
        self.heatmap_generators = [HWHeatmapGenerator(num_joints, stddev)
                                   for stddev in gt_stddevs_pix]
        #
        self.teacher_dir = teacher_output_dir
        #
        # keeps all imgs that have at least one non-crowd annotated person
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]
        if whitelist_ids is not None:
            idset = set(self.ids)
            self.ids = [x for x in whitelist_ids if x in idset]

    def _get_teacher_data(self, img_id, out_hw=None):
        """
        :param as_tensor: We observe that the dataloader runs much faster if
          getitem returns tensor objects.
        """
        if self.teacher_dir is None:
            empty_tensor = torch.zeros(0)
            return (empty_tensor, empty_tensor)
        #
        npz = np.load(os.path.join(self.teacher_dir,
                                   img_id + ".jpg_w48_predictions.npz"))
        # npz["pred_heatmaps"], npz["heatmaps_refined"], npz["heatmaps_order"]
        t_hms = npz["heatmaps_refined"]
        t_ae = npz["embeddings"]
        # npz["heatmaps_refined"], npz["heatmaps_order"]
        t_hms = torch.FloatTensor(t_hms)
        t_ae = torch.FloatTensor(t_ae)
        if out_hw is not None:
            t_hms = torch.nn.functional.interpolate(
                t_hms.unsqueeze(0), out_hw, mode="bilinear",
                align_corners=True)[0]
            t_ae = torch.nn.functional.interpolate(
                t_ae.unsqueeze(0), out_hw, mode="bilinear",
                align_corners=True)[0]

        return t_hms, t_ae

    def get_mask(self, anno, idx):
        coco = self.coco
        img_info = coco.loadImgs(self.ids[idx])[0]
        m = np.zeros((img_info['height'], img_info['width']))
        for obj in anno:
            if obj['iscrowd']:
                rle = pycocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'], img_info['width'])
                m += pycocotools.mask.decode(rle)
            elif obj['num_keypoints'] == 0:
                rles = pycocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'], img_info['width'])
                for rle in rles:
                    m += pycocotools.mask.decode(rle)
        return m < 0.5

    def get_human_segmentation_mask(self, idx):
        """
        Get a boolean mask of same shape as image, where background is false
        and 'person' segmentations are true.
        """
        coco = self.coco
        person_cat_id = coco.getCatIds(catNms=["person"])[0]  # usually 1
        #
        img_info = coco.loadImgs(self.ids[idx])[0]
        mask = np.zeros((img_info['height'], img_info['width']),
                        dtype=np.bool)
        #
        ann_ids = coco.getAnnIds(imgIds=self.ids[idx], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        for a in anns:
            if a["category_id"] == person_cat_id:
                mask |= coco.annToMask(a).astype(np.bool)
        #
        return mask

    def get_joints(self, anno):
        num_people = len(anno)
        joints = np.zeros((num_people, self.num_joints, 3))

        for i, obj in enumerate(anno):
            joints[i, :self.num_joints, :3] = \
                np.array(obj['keypoints']).reshape([-1, 3])
        return joints

    def _do_python_keypoint_eval(self, res_file, res_folder):
        """
        Basically a copypaste of the super method, but added an extra line
        to allow for inline minival via whitelisting: If not specified,
        evaluation will take ALL images into account to average the score.
        To fix this, set the imgIds parameter to just the whitelisted images.
        """
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = pycocotools.cocoeval.COCOeval(self.coco, coco_dt,
                                                  'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.params.imgIds = self.ids  # added this line
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR',
                       'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))
        return info_str

    def __getitem__(self, idx):
        """
        """
        img, anno = super().__getitem__(idx)
        img_id = self.ids[idx]
        img_num = "{:012d}".format(img_id)
        mask = self.get_mask(anno, idx)
        anno = [obj for obj in anno
                if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0]
        joints = self.get_joints(anno)
        hms = [hmg(joints, mask.shape) for hmg in self.heatmap_generators]
        #
        teacher_hms, teacher_ae = self._get_teacher_data(
            img_num, out_hw=mask.shape)
        #
        segm_mask = self.get_human_segmentation_mask(idx)
        # CREATE TENSORS AS IT MAY BE FASTER
        img_id = torch.tensor(img_id, dtype=torch.int64)
        img = torchvision.transforms.functional.to_tensor(img)
        mask = torch.FloatTensor(mask)
        hms = [torch.FloatTensor(hm) for hm in hms]
        segm_mask = torch.FloatTensor(segm_mask)
        #
        return img_id, img, mask, hms, teacher_hms, teacher_ae, segm_mask


class CocoDistillationDatasetAugmented(CocoDistillationDataset):
    """
    This dataset behaves like the non-augmented counterpart, but after
    loading it applies an image-specific transform first (useful e.g. for
    color jittering, grayscaling...), and then an overall transform that
    affects both image and targets (useful e.g. for translation, scale,
    rotation...).
    """

    def __init__(self,
                 coco_root_path,
                 coco_dataset_name,
                 teacher_output_dir=None,
                 remove_images_without_annotations=True,
                 gt_stddevs_pix=[2.0],
                 num_joints=17,
                 img_transform=None,
                 overall_transform=None,
                 whitelist_ids=None):
        """
        """
        super().__init__(coco_root_path, coco_dataset_name, teacher_output_dir,
                         remove_images_without_annotations, gt_stddevs_pix,
                         num_joints, whitelist_ids)
        self.img_transform = img_transform
        self.overall_transform = overall_transform

    def __getitem__(self, idx):
        """
        """
        # all returned vals are FloatTensors
        (img_id, img, mask, hms, teach_hms, teach_ae,
         segm_mask) = super().__getitem__(idx)
        #
        if self.img_transform is not None:
            img = self.img_transform(img)
        #
        if self.overall_transform is not None:
            seed = np.random.randint(2147483647)
            #
            img = torch.cat([self.overall_transform(seed, ch) for ch in img])
            mask = self.overall_transform(seed, mask).squeeze()
            hms = [torch.cat([self.overall_transform(seed, ch) for ch in hm])
                   for hm in hms]
            teach_hms = torch.cat([self.overall_transform(seed, ch)
                                   for ch in teach_hms])
            teach_ae = torch.cat([self.overall_transform(seed, ch)
                                  for ch in teach_ae])
            segm_mask = self.overall_transform(seed, segm_mask).squeeze()
        #
        return img_id, img, mask, hms, teach_hms, teach_ae, segm_mask
