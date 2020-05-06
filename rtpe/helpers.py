# -*- coding:utf-8 -*-


"""
This module contains library utilities for different tasks related to
this library.
"""


import sys
import os
import random
import logging
from socket import gethostname
import datetime
import pytz
#
import torch
import torchvision
# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import coloredlogs  # type: ignore
#
from .third_party.pose_higher_hrnet import PoseHigherResolutionNet
from .third_party.fp16_utils.fp16util import network_to_half


# #############################################################################
# # MODELS
# #############################################################################
def get_hrnet_w48_teacher(w48_statedict_path):
    """
    Instantiates, loads statedict and returnsthe HigherHRNet_w48 from
    https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
    """
    model = PoseHigherResolutionNet(num_joints=17,
                                    tag_per_joint=True,
                                    final_conv_ksize=1,
                                    pretrained_layers=["*"],
                                    inplanes=64,
                                    #
                                    s2_modules=1,
                                    s2_branches=2,
                                    s2_block_type="BASIC",
                                    s2_blocks=[4, 4],
                                    s2_chans=[48, 96],
                                    #
                                    s3_modules=4,
                                    s3_branches=3,
                                    s3_block_type="BASIC",
                                    s3_blocks=[4, 4, 4],
                                    s3_chans=[48, 96, 192],
                                    #
                                    s4_modules=3,
                                    s4_branches=4,
                                    s4_block_type="BASIC",
                                    s4_blocks=[4, 4, 4, 4],
                                    s4_chans=[48, 96, 192, 384],
                                    #
                                    deconvs=1,
                                    deconv_chans=[48],
                                    deconv_ksize=[4],
                                    deconv_num_blocks=4,
                                    deconv_cat=[True],
                                    #
                                    with_ae_loss=(True, False))
    # load_state_dict crashes if not HALF_PRECISION
    model = network_to_half(model)
    model.load_state_dict(torch.load(w48_statedict_path), strict=True)
    model.eval()
    #
    return model


class ModuleSummary:
    """
    This class acts more or less like a functor that given a
    ``torch.nn.Module`` returns its type and number of layers as well as
    number of parameters. Check the ``get_model_summary`` help for more
    details.
    """

    # Change this if the summary info is extended
    COL_NAMES = ("Layer name", "Layer running count", "Num. params.",
                 "Param. shapes")

    def __init__(self):
        """
        """
        self._summlist = []
        self._overall_stats = {"TOTAL PARAMS": 0, "NUM LAYERS": {}}
        # flops currently not supported as they require to propagate an input
        # "TOTAL FLOPS": 0}

    def _summarize_module(self, m):
        """
        """
        m_name = m.__class__.__name__
        m_shapes = {}
        m_params = "Not available"
        # m_flops = None
        #
        m_count = self._overall_stats["NUM LAYERS"].get(m_name, 0) + 1
        self._overall_stats["NUM LAYERS"][m_name] = m_count
        #
        try:
            m_params = m.weight.numel()
            m_shapes["weight"] = list(m.weight.shape)
            if m.bias is not None:
                m_params += m.bias.numel()
                m_shapes["bias"] = list(m.bias.shape)
            self._overall_stats["TOTAL PARAMS"] += m_params
        except AttributeError:
            pass
        #
        # this has to match exactly with self.COL_NAMES
        result = (m_name, m_count, m_params, m_shapes)
        self._summlist.append(result)

    def _summarize_model(self, model):
        """
        """
        model.apply(self._summarize_module)
        return self._summlist, self._overall_stats

    @staticmethod
    def summary_list_to_string(model_name, col_names, summary_lst,
                               overall_stats={}, col_sep=10):
        """
        """
        regex = ("{:<%d}" % col_sep) * len(col_names)
        liststr = [regex.format(*col_names)]
        separator = "=" * len(liststr[0])
        liststr.append(separator)
        for msum in summary_lst:
            liststr.append(regex.format(*[str(elt) for elt in msum]))
        #
        liststr.append(separator)

        overall = ", ".join(["{}: {} ". format(k, v)
                             for k, v in overall_stats.items()])
        if overall:
            liststr.append(overall)
        #
        liststr.insert(0, str(model_name))
        liststr.insert(0, separator)
        return "\n".join(liststr)

    @classmethod
    def get_model_summary(cls, model, as_string=False,
                          str_column_separation=25):
        """
        :param model: A ``torch.nn.Module`` instance to be summarized
        :param bool as_string: If true, the summary is returned as a
          pretty-printed string where each column has the given separation
        :returns: Either the tuple ``(summlist, overall)`` which contain
          respectively the per-layer and overall stats, or a pretty-printed
          string with the same information if as_string is true.

        Usage example::
          summ = ModuleSummary.get_model_summary(student, as_string=True)
          print(summ)

        .. note::

          Only ``torch.nn.Module`` instances will be traversed, so make
          sure your model is compliant (e.g. ``self.convs = [c1, c2...]`` will
          not be accessed, replace with ``torch.nn.ModuleList([c1, c2...])``)
        """
        ms = cls()
        summlist, overall = ms._summarize_model(model)
        if as_string:
            model_name = model.__class__.__name__
            strsumm = ms.summary_list_to_string(model_name, ms.COL_NAMES,
                                                summlist, overall,
                                                str_column_separation)
            return strsumm
        else:
            return summlist, overall


# #############################################################################
# # DATA MANAGEMENT
# #############################################################################
def make_rand_minival_split(val_dir, minival_size, extension=".jpg",
                            imgs=None):
    """
    :returns: Two lists ``(minival, rest_val)``, the first with the randomly
      chosen image **basenames** for the minival split, the second with the
      remaining ones. Note that a **list** is made and shuffled, so the
      memory and runtime requirements will behave accordingly.
    Usage example::

      mv100, _ = make_rand_minival_split(
          "/home/a9fb1e/datasets/coco/images/val2017", 100)
      with open("assets/coco_minival2017_100.txt", "w") as f:
          for elt in mv100:
              f.write("{}\n".format(elt))

      with open("assets/coco_minival2017_100.txt", "r") as f:
          MINIVAL_IDS = [int(line.rstrip('.jpg\n')) for line in f]
    """
    imgs = [p for p in os.listdir(val_dir) if p.endswith(extension)]
    random.shuffle(imgs)
    minival = imgs[:minival_size]
    rest_val = imgs[minival_size:]
    return minival, rest_val


# #############################################################################
# # DATA AUGMENTATION
# #############################################################################
class SeededCompose(torchvision.transforms.Compose):
    """
    If this random transform is repeated several times with the same
    seed, it will produce the same results. Useful when the image and
    the targets need to have e.g. the same rotation but can't be
    transformed in the same call due to channel limitations.
    """
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)

    def __call__(self, seed, *args, **kwargs):
        """
        """
        random.seed(seed)
        return super().__call__(*args, **kwargs)


# #############################################################################
# # PLOTTING
# #############################################################################
def plot_arrays(*arrs, share_zoom=True):
    """
    Plots arrays side by side
    :param share_zoom: If true, zooming left will correspondingly zoom right.
    """
    fig, axarr = plt.subplots(1, len(arrs), sharex=share_zoom,
                              sharey=share_zoom)
    for i, a in enumerate(arrs):
        axarr[i].imshow(a)
        fig.tight_layout()
    plt.show()
    input("\n\nPlotting. Press any key to continue...")
    plt.clf()


# #############################################################################
# # LOGGING
# #############################################################################
def make_timestamp(timezone="Europe/Berlin", with_tz_output=True):
    """
    Output example: day, month, year, hour, min, sec, milisecs:
    10_Feb_2018_20:10:16.151
    """
    ts = datetime.datetime.now(tz=pytz.timezone(timezone)).strftime(
        "%d_%b_%Y_%H:%M:%S.%f")[:-3]
    if with_tz_output:
        return "%s(%s)" % (ts, timezone)
    else:
        return ts

class HostnameFilter(logging.Filter):
    """
    This is needed to include hostname into the logger. See::
      https://stackoverflow.com/a/55584223/4511978
    """

    def filter(self, record) -> bool:
        record.hostname = gethostname()
        return True


class ColorLogger:
    """
    This class:

    1. Creates a ``logging.Logger`` with a convenient configuration.
    2. Attaches ``coloredlogs.install`` to it for colored terminal output
    3. Provides some wrapper methods for convenience

    Usage example::

      # create 2 loggers
      cl1 = ColorLogger("term.and.file.logger", "/tmp/test.txt")
      cl2 = ColorLogger("JustTermLogger")
      # use them at wish
      cl1.debug("this is a debugging message")
      cl2.info("this is an informational message")
      cl1.warning("this is a warning message")
      cl2.error("this is an error message")
      cl1.critical("this is a critical message")
    """

    FORMAT_STR = ("%(asctime)s.%(msecs)03d %(hostname)s: %(name)s" +
                  "[%(process)d] %(levelname)s %(message)s")

    def _get_logger(self, logger_name, logfile_path, filemode="a",
                    logging_level=logging.DEBUG):
        """
        :param filemode: In case ``logfile_path`` is given, this specifies the
          output mode (e.g. 'a' for append).

        :returns: a ``logging.Logger`` configured to output all events at level
          ``self.logging_level`` or above into ``sys.stdout`` and (optionally)
          the given ``logfile_path``, if not None.
        """
        # create logger, formatter and filter, and set desired logger level
        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter(self.FORMAT_STR,
                                      datefmt="%Y-%m-%d %H:%M:%S")
        hostname_filter = HostnameFilter()
        logger.setLevel(logging_level)
        # create and wire stdout handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.addFilter(hostname_filter)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        # optionally, create and wire file handler
        if logfile_path is not None:
            # create one handler for print and one for export
            file_handler = logging.FileHandler(logfile_path, filemode)
            file_handler.addFilter(hostname_filter)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        #
        return logger

    def __init__(self, logger_name, logfile_path, filemode="a",
                 logging_level=logging.DEBUG):
        """
        :param logger_name: A process may have several loggers. This parameter
          distinguishes them.
        :param logfile_path: Where to write out. If not given, log just prints
          to CLI but doesn't save to file.
        """
        self.logger: logging.Logger = self._get_logger(logger_name,
                                                       logfile_path, filemode,
                                                       logging_level)
        #
        coloredlogs.install(logger=self.logger,
                            fmt=self.FORMAT_STR,
                            level=logging_level)
        if logfile_path is not None:
            self.info("[{}] Saving log into {}".format(
                self.__class__.__name__, logfile_path))

    # a few convenience wrappers:
    def debug(self, *args, **kwargs) -> None:
        self.logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs) -> None:
        self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs) -> None:
        self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs) -> None:
        self.logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs) -> None:
        self.logger.critical(*args, **kwargs)
