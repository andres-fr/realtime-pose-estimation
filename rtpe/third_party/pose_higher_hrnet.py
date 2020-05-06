# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# Modified by Andres F. R.
# ------------------------------------------------------------------------------

import os
import logging

import torch
import torch.nn as nn


# class none_module(nn.Module):
#     """
#     https://github.com/pytorch/pytorch/issues/30459#issuecomment-597679482
#     """
#     def __init__(self,):
#         super(none_module, self).__init__()
#         self.none_module_property = True

class NoOpModule(nn.Module):
    """
    https://github.com/pytorch/pytorch/issues/30459#issuecomment-597679482
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return args



BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class PoseHigherResolutionNet(nn.Module):
    BLOCK_TYPES = {
        "BASIC": BasicBlock,
        "BOTTLENECK": Bottleneck
    }

    # def __init__(self, cfg):
    def __init__(self, num_joints=17, tag_per_joint=True, final_conv_ksize=1,
                 pretrained_layers=["*"], inplanes=64,
                 #
                 s2_modules=1, s2_branches=2, s2_block_type="BASIC",
                 s2_blocks=[4, 4], s2_chans=[48, 96], s2_fuse_method="SUM",
                 #
                 s3_modules=4, s3_branches=3, s3_block_type="BASIC",
                 s3_blocks=[4, 4, 4], s3_chans=[48, 96, 192],
                 s3_fuse_method="SUM",
                 #
                 s4_modules=3, s4_branches=4, s4_block_type="BASIC",
                 s4_blocks=[4, 4, 4, 4], s4_chans=[48, 96, 192, 384],
                 s4_fuse_method="SUM",
                 #
                 deconvs=1,
                 deconv_chans=[48],
                 deconv_ksize=[4],
                 deconv_num_blocks=4,
                 deconv_cat=[True],
                 #
                 with_ae_loss=(True, False)
    ):
        """
        :param tag_per_joint: If false, associative embeddings have dim=1.
          If true, it seems that the dim equals number of joints (17 for COCO).
          AE paper says 1 is good, but HigherHRNet repo seems to go for 17.
        :param final_conv_ksize: Kernel size of final conv
        :param pretrained_layers: A list only needed by ``init_weights``,
          includes the names of the layers to load from ``pretrained``. if
          first elt is ``"*"`` it loads everything.
        :param inplanes: It seems that this has to be 64.
        :param s2_modules: Number of modules for stage 2. Same with s3, s4.
        :param s2_branches: Number of branches for stage 2. Same with s3, s4.
          Note that the ``blocks`` and ``channels`` parameters must contain
          exactly this many elements. Check
          ``HighResolutionModule._make_branches``.
        :param s2_block_type: Block type for stage 2. Same with s3, s4.
        :param s2_blocks: List with number of blocks per branch (for this
           stage). Same with s3, s4.
        :param s2_chans: List with number of channels per branch (for this
           stage). Same with s3, s4.
        :param s2_fuse_method: unused?
        :param deconvs: Number of deconvolution layers
        :param deconv_chans: List with number of in channels per deconv layer
        :param deconv_ksize: List with kernel size per deconv layer
        :param deconv_num_blocks: Number of basic blocks in the deconv stage.
        :param deconv_cat: List with one boolean per deconv layer. If true, the
          output of the deconv layers will be ``num_chans+ae_dims``, where
          the dimensionality of the associative embeddings is given by
          ``tag_per_joint``.
        :param with_ae_loss: List of booleans that, for the output stage and
          successive deconvolutions, determine if the associative embedding
          channels get concatenated to the heatmap output channels. E.g. for
          ``deconvs=1``, this will have 2 elements.
        """

        super(PoseHigherResolutionNet, self).__init__()

        self.inplanes = inplanes
        # extra = cfg.MODEL.EXTRA

        self.cfg = {"NUM_JOINTS": num_joints, "TAG_PER_JOINT": tag_per_joint,
                    "FINAL_CONV_KSIZE": final_conv_ksize,
                    "PRETRAINED_LAYERS": pretrained_layers,
                    "STAGE2": {
                        "num_modules": s2_modules,
                        "num_branches": s2_branches,
                        "block_cls": self.BLOCK_TYPES[s2_block_type],
                        "num_blocks": s2_blocks,
                        "num_channels": s2_chans,
                        "fuse_method": s2_fuse_method
                    },
                    "STAGE3": {
                        "num_modules": s3_modules,
                        "num_branches": s3_branches,
                        "block_cls": self.BLOCK_TYPES[s3_block_type],
                        "num_blocks": s3_blocks,
                        "num_channels": s3_chans,
                        "fuse_method": s3_fuse_method
                    },
                    "STAGE4": {
                        "num_modules": s4_modules,
                        "num_branches": s4_branches,
                        "block_cls": self.BLOCK_TYPES[s4_block_type],
                        "num_blocks": s4_blocks,
                        "num_channels": s4_chans,
                        "fuse_method": s4_fuse_method
                    },
                    "DECONV": {
                        "num_deconvs": deconvs,
                        "num_channels": deconv_chans,
                        "kernel_size": deconv_ksize,
                        "num_basic_blocks": deconv_num_blocks,
                        "cat_output": deconv_cat
                    }}

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        # self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        # num_channels = self.stage2_cfg['NUM_CHANNELS']
        # block = self.BLOCK_TYPES[self.stage2_cfg['BLOCK']]
        # num_channels = [
        #     num_channels[i] * block.expansion for i in range(len(num_channels))
        # ]
        # self.transition1 = self._make_transition_layer([256], num_channels)
        # self.stage2, pre_stage_channels = self._make_stage(
        #     self.stage2_cfg, num_channels)
        s2_block_expansion = self.cfg["STAGE2"]["block_cls"].expansion
        s2_num_channels = [ch * s2_block_expansion
                           for ch in self.cfg["STAGE2"]["num_channels"]]
        self.transition1 = self._make_transition_layer([256], s2_num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            num_inchannels=s2_num_channels, **self.cfg["STAGE2"])

        # self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        # num_channels = self.stage3_cfg['NUM_CHANNELS']
        # block = self.BLOCK_TYPES[self.stage3_cfg['BLOCK']]
        # num_channels = [
        #     num_channels[i] * block.expansion for i in range(len(num_channels))
        # ]
        # self.transition2 = self._make_transition_layer(
        #     pre_stage_channels, num_channels)
        # self.stage3, pre_stage_channels = self._make_stage(
        #     self.stage3_cfg, num_channels)
        s3_block_expansion = self.cfg["STAGE3"]["block_cls"].expansion
        s3_num_channels = [ch * s3_block_expansion
                           for ch in self.cfg["STAGE3"]["num_channels"]]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       s3_num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            num_inchannels=s3_num_channels, **self.cfg["STAGE3"])

        # self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        # num_channels = self.stage4_cfg['NUM_CHANNELS']
        # block = self.BLOCK_TYPES[self.stage4_cfg['BLOCK']]
        # num_channels = [
        #     num_channels[i] * block.expansion for i in range(len(num_channels))
        # ]
        # self.transition3 = self._make_transition_layer(
        #     pre_stage_channels, num_channels)
        # self.stage4, pre_stage_channels = self._make_stage(
        #     self.stage4_cfg, num_channels, multi_scale_output=False)
        s4_block_expansion = self.cfg["STAGE4"]["block_cls"].expansion
        s4_num_channels = [ch * s4_block_expansion
                           for ch in self.cfg["STAGE4"]["num_channels"]]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       s4_num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            num_inchannels=s4_num_channels, multi_scale_output=False,
            **self.cfg["STAGE4"])

        # self.final_layers = self._make_final_layers(cfg, pre_stage_channels[0])

        ae_dims = num_joints if tag_per_joint else 1
        self.final_layers = self._make_final_layers(
            pre_stage_channels[0],  # input_channels
            num_joints, ae_dims, with_ae_loss, final_conv_ksize,
            deconvs, deconv_chans)

        # self.deconv_layers = self._make_deconv_layers(
        #     cfg, pre_stage_channels[0])
        self.deconv_layers = self._make_deconv_layers(
            pre_stage_channels[0],  # input_channels
            num_joints, ae_dims, with_ae_loss, deconvs, deconv_num_blocks,
            deconv_chans, deconv_ksize, deconv_cat)

        self.num_deconvs = deconvs  # extra.DECONV.NUM_DECONVS
        # self.deconv_config = cfg.MODEL.EXTRA.DECONV
        # self.loss_config = cfg.LOSS  # UNUSED?
        self.pretrained_layers = pretrained_layers  # cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']
        self.deconv_cat = deconv_cat

    # def _make_final_layers(self, cfg, input_channels):
    def _make_final_layers(self, input_channels, num_joints, ae_dims,
                           with_ae_loss, final_conv_ksize, num_deconvs,
                           deconv_chans):
        """
        """
        # dim_tag = cfg.MODEL.NUM_JOINTS if cfg.MODEL.TAG_PER_JOINT else 1
        # extra = cfg.MODEL.EXTRA
        final_layers = []
        # output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
        #     if cfg.LOSS.WITH_AE_LOSS[0] else cfg.MODEL.NUM_JOINTS
        output_channels = num_joints
        if with_ae_loss[0]:
            output_channels += ae_dims
        final_layers.append(nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=final_conv_ksize,  # extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if final_conv_ksize == 3 else 0  # extra.FINAL_CONV_KERNEL == 3 else 0
        ))
        # deconv_cfg = extra.DECONV
        # for i in range(deconv_cfg.NUM_DECONVS):
        for i in range(num_deconvs):
            input_channels = deconv_chans[i]  # deconv_cfg.NUM_CHANNELS[i]
            # output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
            #     if cfg.LOSS.WITH_AE_LOSS[i+1] else cfg.MODEL.NUM_JOINTS
            output_channels = num_joints
            if with_ae_loss[i + 1]:
                output_channels += ae_dims
            final_layers.append(nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=final_conv_ksize,  # extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if final_conv_ksize == 3 else 0  # extra.FINAL_CONV_KERNEL == 3 else 0
            ))
        return nn.ModuleList(final_layers)

    def _make_deconv_layers(self, input_channels, num_joints, ae_dims,
                            with_ae_loss, num_deconvs, deconv_num_blocks,
                            deconv_chans, deconv_ksize, deconv_cat):
        # dim_tag = cfg.MODEL.NUM_JOINTS if cfg.MODEL.TAG_PER_JOINT else 1
        # extra = cfg.MODEL.EXTRA
        # deconv_cfg = extra.DECONV
        deconv_layers = []
        # for i in range(deconv_cfg.NUM_DECONVS):
        #     if deconv_cfg.CAT_OUTPUT[i]:
        #         final_output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
        #             if cfg.LOSS.WITH_AE_LOSS[i] else cfg.MODEL.NUM_JOINTS

        for i in range(num_deconvs):
            if deconv_cat[i]:
                # final_output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
                #     if cfg.LOSS.WITH_AE_LOSS[i] else cfg.MODEL.NUM_JOINTS
                final_output_channels = num_joints
                if with_ae_loss[i]:
                    final_output_channels += ae_dims
                input_channels += final_output_channels
            # output_channels = deconv_cfg.NUM_CHANNELS[i]
            output_channels = deconv_chans[i]
            # deconv_kernel, padding, output_padding = \
            #     self._get_deconv_cfg(deconv_cfg.KERNEL_SIZE[i])
            deconv_kernel, padding, output_padding = \
                self._get_deconv_cfg(deconv_ksize[i])

            layers = []
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False),
                nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ))
            # for _ in range(cfg.MODEL.EXTRA.DECONV.NUM_BASIC_BLOCKS):
            for _ in range(deconv_num_blocks):
                layers.append(nn.Sequential(
                    BasicBlock(output_channels, output_channels),
                ))
            deconv_layers.append(nn.Sequential(*layers))
            input_channels = output_channels

        return nn.ModuleList(deconv_layers)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    # transition_layers.append(None)
                    transition_layers.append(NoOpModule())
                    ### transition_layers.append(none_module())
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self,
                    # layer_config,
                    num_modules, num_branches, num_blocks, num_channels,
                    block_cls, fuse_method,
                    num_inchannels,
                    multi_scale_output=True):
        # num_modules = layer_config['NUM_MODULES']
        # num_branches = layer_config['NUM_BRANCHES']
        # num_blocks = layer_config['NUM_BLOCKS']
        # num_channels = layer_config['NUM_CHANNELS']
        # block = self.BLOCK_TYPES[layer_config['BLOCK']]
        # fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    # block,
                    block_cls,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        # for i in range(self.stage2_cfg['NUM_BRANCHES']):
        for i in range(self.cfg["STAGE2"]["num_branches"]):
            x_list.append(self.transition1[i](x))  #
        y_list = self.stage2(x_list)
        x_list = []


        # for i in range(self.stage3_cfg['NUM_BRANCHES']):
        for i in range(self.cfg["STAGE3"]["num_branches"]):
            if not isinstance(self.transition2[i], NoOpModule):
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)


        x_list = []
        # for i in range(self.stage4_cfg['NUM_BRANCHES']):
        for i in range(self.cfg["STAGE4"]["num_branches"]):
            if not isinstance(self.transition3[i], NoOpModule):
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        final_outputs = []
        x = y_list[0]
        y = self.final_layers[0](x)
        final_outputs.append(y)

        for i in range(self.num_deconvs):
            # if self.deconv_config.CAT_OUTPUT[i]:
            if self.deconv_cat[i]:
                x = torch.cat((x, y), 1)

            x = self.deconv_layers[i](x)
            y = self.final_layers[i+1](x)
            final_outputs.append(y)

        return final_outputs

    def init_weights(self, pretrained='', verbose=True):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    if name in parameters_names or name in buffers_names:
                        if verbose:
                            logger.info(
                                '=> init {} from {}'.format(name, pretrained)
                            )
                        need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)

    # def forward_stem(self, x):
    #     """
    #     """
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.conv2(x)
    #     x = self.bn2(x)
    #     x = self.relu(x)
    #     x = self.layer1(x)
    #     return x
