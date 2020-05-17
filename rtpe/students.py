# -*- coding:utf-8 -*-


"""
This module contains the different components and architectures that are used
to instantiate the student neural networks.
"""


import torch
#
from .third_party.pose_higher_hrnet import BN_MOMENTUM, Bottleneck
from .third_party.fp16_utils.fp16util import network_to_half
from .third_party.RSB import ResidualStepBlock


# #############################################################################
# # WEIGHT INITIALIZATION
# #############################################################################
def init_weights(module, init_fn=torch.nn.init.kaiming_normal,
                 bias_val=0.0):
    """
    """
    if isinstance(module, torch.nn.Linear):
        init_fn(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(bias_val)
    elif isinstance(module, torch.nn.Conv2d):
        init_fn(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(bias_val)


# #############################################################################
# # BOTTLENECKS
# #############################################################################
class SkipConv(torch.nn.Module):
    """
    A Bottleneck inspired from HigherHRNet paper/software and from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, in_chans, out_chans, ksizes,
                 strides=None, dilations=None, paddings=None,
                 downsample=None, bn_momentum=0.1):
        """
        """
        super().__init__()
        #
        if strides is None:
            strides = [1 for _ in in_chans]
        if dilations is None:
            dilations = [1 for _ in in_chans]
        if paddings is None:
            paddings = [0 for _ in in_chans]
        assert len(in_chans) == len(out_chans) == len(ksizes) == \
            len(strides) == len(dilations) == len(paddings), \
            "Channels, ksizes, strides and dilations must be of same length!"
        #
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s,
                             dilation=d, padding=p, bias=False)
             for (in_ch, out_ch, k, s, d, p) in zip(in_chans, out_chans,
                                                    ksizes, strides, dilations,
                                                    paddings)])
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm2d(out_ch, momentum=bn_momentum)
             for out_ch in out_chans])
        self.relus = torch.nn.ModuleList([torch.nn.ReLU(inplace=True)
                                          for _ in out_chans])
        #
        self.downsample = downsample
        self.final_relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        """
        """
        if self.downsample is not None:
            residual = self.downsample(x)
        # regular forward prop
        # print("skipconv x shape:", x.shape)
        for conv, bn, relu in zip(self.convs, self.bns, self.relus):
            x = conv(x)
            x = bn(x)
            x = relu(x)
            # print(" >>> skipconv x shape:", x.shape)
        #
        x = x + residual
        x = self.final_relu(x)
        return x


def get_straight_skip_conv(in_chans, out_chans, bn_momentum=0.1):
    """
    """
    num_in = len(in_chans)
    num_out = len(out_chans)
    assert num_in == num_out, "in_chans and out_chans must have same length!"
    ksizes = [3 for _ in range(num_in)]
    paddings = [1 for _ in range(num_in)]
    strides = [1 for _ in range(num_in)]
    dilations = [1 for _ in range(num_in)]
    #
    downsample = torch.nn.Sequential(
        torch.nn.Conv2d(in_chans[0], out_chans[-1], kernel_size=1,
                        stride=1, padding=0, bias=False),
        # torch.nn.AvgPool2d([2, 2], stride=[1, 1], padding=0),
        torch.nn.BatchNorm2d(out_chans[-1], momentum=bn_momentum))
    #
    skc = SkipConv(in_chans, out_chans, ksizes, strides, dilations,
                   paddings, downsample, bn_momentum)
    return skc


# #############################################################################
# # PCR PAPER
# #############################################################################
class SELayer(torch.nn.Module):
    """
    Squeeze-excitation module
    """
    def __init__(self, in_chans, hidden_chans=None, bn_momentum=0.1):
        super().__init__()
        if hidden_chans is None:
            hidden_chans = in_chans // 4
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)  # output a scalar per ch
        self.fc = torch.nn.Sequential(
            # BN doesn't seem to work with linear, so we add a bias instead
            torch.nn.Linear(in_chans, hidden_chans, bias=True),
            # torch.nn.BatchNorm2d(hidden_chans, momentum=bn_momentum),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_chans, in_chans, bias=True),
            # torch.nn.BatchNorm2d(hidden_chans, momentum=bn_momentum),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # return x * y.expand_as(x)
        return y


class ContextAwareModule(torch.nn.Module):
    """
    """

    def __init__(self, in_chans, se_chans=None,
                 hdc_dilations=[1, 2, 3, 4],  # hdc_ksize=3, hdc_stride=1,
                 hdc_chans=None, bn_momentum=0.1):
        """
        """
        super().__init__()
        # residual branch
        self.residual = torch.nn.Sequential(
            torch.nn.Conv2d(in_chans, in_chans, kernel_size=1,
                            stride=1, bias=False),
            torch.nn.BatchNorm2d(in_chans, momentum=bn_momentum),
            torch.nn.ReLU(inplace=True))
        # squeeze excitation branch
        self.se = SELayer(in_chans, se_chans, bn_momentum)
        # hybrid dilated conv branch (hdcs need to be concatenated)
        if hdc_chans is None:
            hdc_chans = in_chans // 4
        hdc_top_chans = hdc_chans * len(hdc_dilations)
        self.hdcs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(in_chans, hdc_chans, kernel_size=3,
                                stride=1, dilation=d, padding=d,
                                bias=False),
                torch.nn.BatchNorm2d(hdc_chans, momentum=bn_momentum),
                torch.nn.ReLU(inplace=True))
            for d in hdc_dilations])
        self.hdc_top = torch.nn.Sequential(
            torch.nn.Conv2d(hdc_top_chans, in_chans, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(in_chans, momentum=bn_momentum),
            torch.nn.ReLU(inplace=True))
        # final ReLU
        self.final_relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        """
        """
        _, _, in_h, in_w = x.shape
        # residual branch
        residual = self.residual(x)
        # squeeze excitation branch
        attention = self.se(x)
        # hybrid dilated conv branch
        out = torch.cat([hdc(x) for hdc in self.hdcs], dim=1)
        out = self.hdc_top(out)
        # optionally upsample HDC
        _, _, h, w = out.shape
        if (h != in_h) or (w != in_w):
            out = torch.nn.functional.interpolate(
                out, (in_h, in_w), mode="bilinear", align_corners=True)
        #
        out = residual + (out * attention.expand_as(out))
        out = self.final_relu(out)
        return out

# #############################################################################
# # HHRNET STEM
# #############################################################################
class StemHRNet(torch.nn.Module):
    """
    # HHRNET: params=63,827,139, GFLOPS=154,3
    # Stem: params=325,056, GFLOPS=8,355
    """
    INPLANES = 64

    def __init__(self):
        """
        Stem snippet copypasted from the HigherHRNet constructor.
        """
        super().__init__()
        #
        self.conv1 = torch.nn.Conv2d(
            3, self.INPLANES, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.INPLANES, momentum=BN_MOMENTUM)
        self.conv2 = torch.nn.Conv2d(
            self.INPLANES, self.INPLANES, kernel_size=3, stride=2,
            padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(self.INPLANES, momentum=BN_MOMENTUM)
        self.relu = torch.nn.ReLU(inplace=True)
        #
        self.layer1 = self._make_layer1(self.INPLANES, 4)

    def _make_layer1(self, planes=64, blocks=4):
        """
        Stem snippet copypasted/refactored from the HigherHRNet constructor.
        """
        block_cls = Bottleneck
        expansion = block_cls.expansion
        #
        residual_fn = torch.nn.Sequential(
            torch.nn.Conv2d(self.INPLANES, planes * expansion,
                            kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(planes * expansion, momentum=BN_MOMENTUM))
        #
        layers = []
        layers.append(block_cls(64, 64, 1, residual_fn))
        #
        for i in range(1, blocks):
            layers.append(block_cls(planes * expansion, planes))
        #
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Stem snippet copypasted from the HigherHRNet forward method.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        return x

    def load_pretrained(self, hhrnet_statedict_path, device="cpu",
                        check=False):
        """
        Loads the HigherHRNet parameters that correspond to the stem part.
        To load it successfully in half precision, the following works::

          stem = network_to_half(StemHRNet())
          stem[1].load_pretrained(hhrnet_statedict_path, device, check=False)

        Also note that half precision seems to work on ``cuda`` only.
        """
        hhrnet_d = torch.load(hhrnet_statedict_path, map_location=device)
        filtered_dict = {k: hhrnet_d["1." + k] for k in self.state_dict()}
        #
        self.load_state_dict(filtered_dict)
        #
        if check:
            assert all(
                [(hhrnet_d["1." + k].to(device) == v.to(device)).all() for k, v
                 in self.state_dict().items()]), "Error loading statedict!"


def get_pretrained_stem(hhrnet_statedict_path, device="cuda",
                        half_precision=True):
    """
    """
    if half_precision:
        stem = network_to_half(StemHRNet())
    else:
        stem = torch.nn.Sequential(torch.nn.Identity(), StemHRNet())
    stem[1].load_pretrained(hhrnet_statedict_path, device,
                            check=False)
    return stem


# #############################################################################
# # MODELS
# #############################################################################

class RefinerStudent(torch.nn.Module):
    """
    """
    def __init__(self, hhrnet_statedict_path=None, device="cuda",
                 layers_per_stage=[3, 3, 3],
                 num_heatmaps=17, ae_dims=1,
                 half_precision=True, init_fn=torch.nn.init.kaiming_normal_,
                 trainable_stem=False, bn_momentum=0.1):
        """
        """
        super().__init__()
        self.bn_momentum = bn_momentum
        self.layers_per_stage = layers_per_stage
        self.num_heatmaps = num_heatmaps
        self.ae_dims = ae_dims
        # load stem
        self.stem = StemHRNet()
        self.stem_out_chans = self.stem.layer1[-1].bn3.num_features
        self.trainable_stem = trainable_stem
        #
        # load body
        self.stages = self._make_body()
        # optionally initialize parameters
        if init_fn is not None:
            self.apply(lambda module: init_weights(module, init_fn, 0.0))
        if half_precision:
            self.stem = network_to_half(self.stem)
        else:
            self.stem = torch.nn.Sequential(torch.nn.Identity(), self.stem)
        if hhrnet_statedict_path is not None:
            self.stem[1].load_pretrained(
                hhrnet_statedict_path, device, check=False)
        #
        self.to(device)

    def save_body(self, out_path):
        """
        """
        torch.save(self.stages.state_dict(), out_path)

    def load_body(self, statedict_path):
        """
        """
        self.stages.load_state_dict(torch.load(statedict_path))

    def _make_body(self):
        """
        """
        stages = torch.nn.ModuleList()
        ch = self.stem_out_chans
        # add all stages but last with same number of channels
        for l in self.layers_per_stage[:-1]:
            in_chans = [ch for _ in range(l)]
            out_chans = [ch for _ in range(l)]
            stage = get_straight_skip_conv(in_chans, out_chans,
                                           self.bn_momentum)
            stages.append(stage)
        # last stage outputs num_heatmaps + ae_dims
        last_l = self.layers_per_stage[-1]
        in_chans = [ch for _ in range(last_l)]
        out_chans = [ch for _ in range(last_l)]
        out_chans[-1] = self.num_heatmaps + self.ae_dims
        stage = get_straight_skip_conv(in_chans, out_chans,
                                       self.bn_momentum)
        stages.append(stage)
        #
        return stages  # torch.nn.Sequential(*stages)

    def forward(self, x, out_hw=None):
        """
        """
        if self.trainable_stem:
            stem_out = self.stem(x)
        else:
            with torch.no_grad():
                stem_out = self.stem(x)
        # progressive context refinement
        x = self.stages[0](stem_out)
        for s in self.stages[1:]:
            x = s(stem_out + x)
        # resize last stage
        if out_hw is not None:
            x = torch.nn.functional.interpolate(
                x, out_hw, mode="bilinear", align_corners=True)
        return x


class MultistageStudent(RefinerStudent):
    """
    Basic idea:
    1. Replace bottlenecks with step blocks
    2. Introduce squeeze-excitation among stages
    3. figure out how to do intermediate supervision with decreasing stddev.
    """
    REMARKS = "Second attempt. We added intermediate supervision"

    def __init__(self, hhrnet_statedict_path=None, device="cuda",
                 layers_per_stage=[3, 3, 3],
                 num_heatmaps=17, ae_dims=1,
                 half_precision=True, init_fn=torch.nn.init.kaiming_normal_,
                 trainable_stem=False, bn_momentum=0.1):
        """
        """
        super().__init__()
        self.bn_momentum = bn_momentum
        self.layers_per_stage = layers_per_stage
        self.num_heatmaps = num_heatmaps
        self.ae_dims = ae_dims
        # load stem
        self.stem = StemHRNet()
        self.stem_out_chans = self.stem.layer1[-1].bn3.num_features
        self.trainable_stem = trainable_stem
        #
        # load body
        self.stages = self._make_body()
        # optionally initialize parameters
        if init_fn is not None:
            self.apply(lambda module: init_weights(module, init_fn, 0.0))
        if half_precision:
            self.stem = network_to_half(self.stem)
        else:
            self.stem = torch.nn.Sequential(torch.nn.Identity(), self.stem)
        if hhrnet_statedict_path is not None:
            self.stem[1].load_pretrained(
                hhrnet_statedict_path, device, check=False)
        #
        self.to(device)

    def save_body(self, out_path):
        """
        """
        torch.save(self.stages.state_dict(), out_path)

    def load_body(self, statedict_path):
        """
        """
        self.stages.load_state_dict(torch.load(statedict_path))

    def _make_body(self):
        """
        """
        stages = torch.nn.ModuleList()
        stem_ch = self.stem_out_chans
        out_ch = self.num_heatmaps + self.ae_dims
        #
        for stage_i, l in enumerate(self.layers_per_stage):
            in_chans = [out_ch + stem_ch for _ in range(l)]
            out_chans = [out_ch + stem_ch for _ in range(l)]
            # first input of first stage gets stem_ch only
            if stage_i == 0:
                in_chans[0] = stem_ch
            # last output of all stages is out_ch only
            out_chans[-1] = out_ch
            #
            ksizes = [3 for _ in range(l)]
            paddings = [1 for _ in range(l)]
            strides = [1 for _ in range(l)]
            dilations = [1 for _ in range(l)]
            # downsample from in chans[0] to out chans[-1] is always needed
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_chans[0], out_chans[-1], kernel_size=1,
                                stride=1, padding=0, bias=False),
                # torch.nn.AvgPool2d([2, 2], stride=[1, 1], padding=0),
                torch.nn.BatchNorm2d(out_chans[-1],
                                     momentum=self.bn_momentum))
            #
            stage = SkipConv(in_chans, out_chans, ksizes, strides, dilations,
                             paddings, downsample, self.bn_momentum)
            stages.append(stage)
        return stages

    def forward(self, x, out_hw=None):
        """
        """
        if self.trainable_stem:
            stem_out = self.stem(x)
        else:
            with torch.no_grad():
                stem_out = self.stem(x)
        if out_hw is not None:
            stem_out = torch.nn.functional.interpolate(
                stem_out, out_hw, mode="bilinear", align_corners=True)
        #
        stage_out1 = self.stages[0](stem_out)
        if out_hw is not None:
            stage_out1 = torch.nn.functional.interpolate(
                stage_out1, out_hw, mode="bilinear", align_corners=True)
        #
        stage_outs = [stage_out1]
        for s in self.stages[1:]:
            # stack the stage_outs[-1] with stem_out and feed to s
            st_out = s(torch.cat([stem_out, stage_outs[-1]], dim=1))
            if out_hw is not None:
                st_out = torch.nn.functional.interpolate(
                    st_out, out_hw, mode="bilinear", align_corners=True)
            stage_outs.append(st_out)
        return stage_outs



class CamStudent(torch.nn.Module):
    """
    """
    def __init__(self, hhrnet_statedict_path=None, device="cuda",
                 inplanes=48,
                 num_stages=3,
                 num_heatmaps=17, ae_dims=1,
                 half_precision=True, init_fn=torch.nn.init.kaiming_normal_,
                 trainable_stem=False, bn_momentum=0.1):
        """
        """
        super().__init__()
        self.bn_momentum = bn_momentum
        self.num_stages = num_stages
        self.num_heatmaps = num_heatmaps
        self.ae_dims = ae_dims
        # load stem
        self.stem = StemHRNet()
        self.stem_out_chans = self.stem.layer1[-1].bn3.num_features
        self.trainable_stem = trainable_stem
        # mid stem
        self.inplanes = inplanes
        self.mid_stem = torch.nn.Sequential(
            torch.nn.Conv2d(self.stem_out_chans, inplanes, kernel_size=3,
                            stride=1, dilation=1, padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(inplanes, momentum=bn_momentum),
            torch.nn.ReLU(inplace=True))
        # load body
        self.cams, self.hm_convs = self._make_body()
        # optionally initialize parameters
        if init_fn is not None:
            self.apply(lambda module: init_weights(module, init_fn, 0.0))
        if half_precision:
            self.stem = network_to_half(self.stem)
        else:
            self.stem = torch.nn.Sequential(torch.nn.Identity(), self.stem)
        if hhrnet_statedict_path is not None:
            self.stem[1].load_pretrained(
                hhrnet_statedict_path, device, check=False)
        #
        self.to(device)

    def save_body(self, out_path):
        """
        """
        torch.save(self.stages.state_dict(), out_path)

    def load_body(self, statedict_path):
        """
        """
        self.stages.load_state_dict(torch.load(statedict_path))

    def _make_body(self):
        """
        """
        hm_out_ch = self.num_heatmaps + self.ae_dims
        #
        cams = torch.nn.ModuleList()
        hms = torch.nn.ModuleList()
        for i in range(self.num_stages):
            cams.append(ContextAwareModule(self.inplanes, hdc_dilations=[1, 2, 3, 5, 8, 12]))
            hms.append(torch.nn.Conv2d(self.inplanes, hm_out_ch,
                                       kernel_size=3, padding=1, bias=True))
        return cams, hms

    def forward(self, x, out_hw=None, return_intermediate=False):
        """
        """
        out_hms = []
        if self.trainable_stem:
            stem_out = self.stem(x)
            stem_out = self.mid_stem(stem_out)
        else:
            with torch.no_grad():
                stem_out = self.stem(x)
                stem_out = self.mid_stem(stem_out)
        # progressive context refinement
        if return_intermediate:
            raise NotImplementedError
        else:
            x = self.cams[0](stem_out)
            for cam in self.cams[1:]:
                x = x + cam(stem_out)
            out = [self.hm_convs[-1](x)]
        # finally reshape if needed
        if out_hw is not None:
            out = [torch.nn.functional.interpolate(
                x, out_hw, mode="bilinear", align_corners=True)
                   for x in out]
        return out


class AttentionStudent(torch.nn.Module):
    """
    """
    def __init__(self, hhrnet_statedict_path=None, device="cuda",
                 inplanes=48,
                 # num_stages=3,
                 num_heatmaps=17, ae_dims=1,
                 half_precision=True, init_fn=torch.nn.init.kaiming_normal_,
                 trainable_stem=False, bn_momentum=0.1):
        """
        """
        super().__init__()
        self.bn_momentum = bn_momentum
        # self.num_stages = num_stages
        self.num_heatmaps = num_heatmaps
        self.ae_dims = ae_dims
        # load stem
        self.stem = StemHRNet()
        self.stem_out_chans = self.stem.layer1[-1].bn3.num_features
        self.trainable_stem = trainable_stem
        # mid stem
        self.inplanes = inplanes
        mid_inplanes = (self.stem_out_chans + self.inplanes) // 2
        self.mid_stem = torch.nn.Sequential(
            torch.nn.Conv2d(self.stem_out_chans, mid_inplanes, kernel_size=3,
                            stride=1, dilation=1, padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(mid_inplanes, momentum=bn_momentum),
            torch.nn.ReLU(inplace=True),
            #
            torch.nn.Conv2d(mid_inplanes, inplanes, kernel_size=3,
                            stride=1, dilation=1, padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(inplanes, momentum=bn_momentum),
            torch.nn.ReLU(inplace=True))
        # load body:
        self.att_lo, self.att_mid, self.att_hi, self.att_top = self._attention_body()
        self.det_lo, self.det_mid, self.det_hi, self.det_top = self._detection_body_v1()
        # self.det = self._detection_body()
        # self.cams, self.hm_convs = self._make_body()
        # optionally initialize parameters
        if init_fn is not None:
            self.apply(lambda module: init_weights(module, init_fn, 0.0))
        if half_precision:
            self.stem = network_to_half(self.stem)
        else:
            self.stem = torch.nn.Sequential(torch.nn.Identity(), self.stem)
        if hhrnet_statedict_path is not None:
            self.stem[1].load_pretrained(
                hhrnet_statedict_path, device, check=False)
        #
        self.to(device)
        self.device = device

    def _attention_body(self):
        """
        """
        # 1. send stem to an avg pool
        low_res = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1,
                               count_include_pad=False),
            ContextAwareModule(self.inplanes, hdc_dilations=[1, 2, 3, 4, 5]),
        )
        #
        mid_res = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1,
                               count_include_pad=False),
            ContextAwareModule(self.inplanes, hdc_dilations=[1, 2, 3, 4, 5]),
        )
        #
        high_res = torch.nn.Sequential(
            ContextAwareModule(self.inplanes, hdc_dilations=[1, 2, 3, 4, 5]))
        #
        top = torch.nn.Sequential(
            torch.nn.Conv2d(self.inplanes, 1, kernel_size=3,
                            stride=1, dilation=1, padding=1,
                            bias=True),
            # torch.nn.Sigmoid()  # not needed because loss is BCE with logits
        )
        #
        return torch.nn.ModuleList([low_res, mid_res, high_res, top])

    def _detection_body_v1(self):
        """
        """
        hm_out_ch = self.num_heatmaps + self.ae_dims
        #
        low_res = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1,
                               count_include_pad=False),
            ContextAwareModule(self.inplanes, hdc_dilations=[1, 2, 3, 4]),
        )
        #
        mid_res = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1,
                               count_include_pad=False),
            ContextAwareModule(self.inplanes, hdc_dilations=[1, 2, 3, 4]),
            # torch.nn.Upsample(scale_factor=2, mode="nearest")
        )
        #
        high_res = torch.nn.Sequential(
            ContextAwareModule(self.inplanes, hdc_dilations=[1, 2, 3, 4]))
        #
        top = torch.nn.Sequential(
            torch.nn.Conv2d(self.inplanes, hm_out_ch, kernel_size=3,
                            stride=1, dilation=1, padding=1,
                            bias=True))
        #
        return torch.nn.ModuleList([low_res, mid_res, high_res, top])

    def load_state_dicts(self, inpath):
        """
        :param inpath: something like ``os.path.join(SNAPSHOT_DIR,
          "{}_epoch{}_step{}".format(LOAD_TIMESTAMP, LOAD_EPOCH, LOAD_STEP))``
        """
        self.mid_stem.load_state_dict(torch.load(
            inpath + "mid_stem.statedict", map_location=self.device))
        self.att_lo.load_state_dict(torch.load(
            inpath + "att_lo.statedict", map_location=self.device))
        self.att_mid.load_state_dict(torch.load(
            inpath + "att_mid.statedict", map_location=self.device))
        self.att_hi.load_state_dict(torch.load(
            inpath + "att_hi.statedict", map_location=self.device))
        self.att_top.load_state_dict(torch.load(
            inpath + "att_top.statedict", map_location=self.device))

    def forward(self, x, out_hw=None, return_intermediate=False):
        """
        """
        out_hms = []
        if self.trainable_stem:
            stem_out = self.stem(x)
            stem_out = self.mid_stem(stem_out)
        else:
            with torch.no_grad():
                stem_out = self.stem(x)
                stem_out = self.mid_stem(stem_out)
        # human mask detector
        hi = self.att_hi(stem_out)
        mid = self.att_mid(stem_out)
        lo = self.att_lo(mid)
        mid = torch.nn.functional.interpolate(lo, stem_out.shape[-2:],
                                              mode="nearest")
        lo = torch.nn.functional.interpolate(lo, stem_out.shape[-2:],
                                             mode="nearest")
        att = hi + mid + lo
        att = self.att_top(att)

        # att = torch.nn.functional.softmax(att) * att.shape[-1] * att.shape[-2]
        # att = self.att_top(torch.cat((lo, hi), dim=1))
        # keypoint detector

        ### THIS WAS THE ORIGINAL ATTENTION APPROACH
        # att = torch.nn.functional.sigmoid(att / 20)
        #  stem_out = stem_out * att.expand(stem_out.shape)

        ### LOSS WITH THIS REDUCES FASTER, BUT KEYPOINTS STILL NOISY AF.
        att = torch.nn.functional.sigmoid(att / 20)
        stem_out = stem_out + att.expand(stem_out.shape)


        hi = self.det_hi(stem_out)
        mid = self.det_hi(stem_out)
        lo = self.det_lo(mid)
        mid = torch.nn.functional.interpolate(lo, stem_out.shape[-2:],
                                              mode="nearest")
        lo = torch.nn.functional.interpolate(lo, stem_out.shape[-2:],
                                             mode="nearest")
        # lo = torch.nn.functional.interpolate(lo, hi.shape[-2:], mode="nearest")
        # det = self.det_top(torch.cat((lo, hi), dim=1))
        det = hi + mid + lo
        det = self.det_top(det)
        #
        return att, det
        #     x = self.cams[0](stem_out)
        #     for cam in self.cams[1:]:
        #         x = x + cam(stem_out)
        #     out = [self.hm_convs[-1](x)]
        # # finally reshape if needed
        # if out_hw is not None:
        #     out = [torch.nn.functional.interpolate(
        #         x, out_hw, mode="bilinear", align_corners=True)
        #            for x in out]
        # return out
