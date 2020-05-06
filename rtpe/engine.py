# -*- coding:utf-8 -*-


"""
This module contains app-related tasks to make the training/evaluation loops
and other usual tasks less verbose.
"""


import os
import matplotlib
#
from .third_party.vis import save_valid_image
#
from .helpers import plot_arrays


# #############################################################################
# #
# #############################################################################
def eval_student(model, hm_parser, val_dataloader, device,
                 plot_every=None, save_every=None, save_dir="/tmp"):
    """
    :param model: A ``torch.nn.Module`` instance that accepts a batch of images
     and returns a single tensor of predictions.
    :param hm_parser: An instance of the third party's ``HeatmapParser``
    :param val_dataloader: An instance of ``CocoDistillationDataset`` holding
      the data to evaluate the model on.
    :returns: ``eval_dict``, a dict with the evaluation names and results.

    For each image in the val dataloader, the function runs the model and the
    hm_parser (optionally plotting/saving the results), and finally computes,
    prints and returns the evaluation on the COCO official metrics.
    """
    model.eval()
    all_preds = []
    all_scores = []
    #
    for batch_i, (img_id, img, mask, hms, _, _) in enumerate(val_dataloader):
        print("eval:", batch_i)
        out_hw = img.shape[2:]
        img = img.to(device)
        pred = model(img, out_hw)
        pred = pred.cpu().detach()
        pred_hms = pred[:, :17]
        pred_ae = pred[:, 17:]
        # parser needs hms(1, 17, h, w) and ae (1, AE_DIM, h, w, 1)
        grouped, scores = hm_parser.parse(pred_hms, pred_ae.unsqueeze(-1),
                                          adjust=True, refine=True)
        # for evaluation
        final_results = [x for x in grouped[0] if x.size > 0]
        all_preds.append(final_results)
        all_scores.append(scores)
        # save predictions
        img = img[0].cpu()
        if save_every is not None and batch_i % save_every == 0:
            save_valid_image(
                img.sub(img.min()).mul(255.0 / img.max()).permute(
                    1, 2, 0).numpy(),
                [x for x in grouped[0] if x.size > 0],
                os.path.join(save_dir, "student_minival_{}.jpg".format(batch_i)),
                dataset="COCO")
        # plot predictions
        if plot_every is not None and batch_i % plot_every == 0:
            matplotlib.use("TkAgg")
            plot_arrays(img.permute(1, 2, 0),
                        *[hm[0].sum(dim=0) for hm in hms],
                        pred_hms[0].sum(dim=0),
                        pred_ae[0].mean(dim=0))
    #
    eval_dict, mAP = val_dataloader.dataset.evaluate(
        all_preds, all_scores, ".", False, False)
    eval_str = "\n".join([k+"="+str(v) for k, v in eval_dict.items()])
    print(eval_str)
    return eval_dict
