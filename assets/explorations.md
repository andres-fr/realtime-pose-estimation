# EXPLORATIONS


## Changelog:

### May 17

* After few "raw" attempts, redesigned the student and dataloader to predict human segmentation masks. After a few tweaks, the `AttentionStudent` was able to capture both augmented train data and val with success.

The results seem to show that the capacity is just OK for segm. masks (in terms of receptive field and resolution). attention may benefit from adding a few more maps to trim false positives, and also to avoid competing with the keypoint detector when modifying the mid stem.

Detection seems to be in big need of capacity enhancement. Observing the results, it seems that the detector has several issues:

1. It can't provide the spatial resolution needed
2. All maps are similar: It can't distinguish them, particularly given the addition of the attention part.

### May 18

Increased capacity of keypoint pipeline using step nets. After training for a while, attention learned OK but KPs not. So Increased KP a little more, removed mid stem from attention optimizer, and passed KP detections through sigmoid(det/20) to try to emulate the success of the attention part.

in 3 hours, the KP loss went from 0.8 to 0.72 while attention loss maintained. Maps look better, also in validation5000. So it seems that the attention part doesn't need to train the stem at this point.

Since the attention has much more fpos than fneg, we go back to the multiplication schema. The network seems to be lacking a lot of spatial resolution and details, so we scale up the stem (after mult. by attention) and we concat the img before feeding the keypoint detector.

Training interrupted after 7 epochs (2h) for non-related reasons. Very steep improvement in detection loss from 0.7 to 0.2, but still no visible keypoints in the output.

### May 20

Last run was promising, so resumed BUT replaced the concatenated img with the LAB color space version. Everything else identical.
Result: Attention train loss maintained, but eval attention got visibly better. so training longer after convergence helped.
Detection converged at 0.2. Problem: all detection maps seem equal! No traces of proper heatmaps.

##### 20_May_2020_15:05:54.922:
* Concat the img also before the attention "top" layer, to try to achieve more precise segmentations, crf-like. Also reduce capacity of attention part (eliminated the 5th stride), we need it more in detection.
* Maybe S-E is hurting detection? Added an extra CAM, and duplicated channels for SE and HDC bottlenecks.
* Reduced capacity of attention pipeline and inplanes to 100. Reduced batch size from 10 to 8.
* Detection CAMs now perform progressive refinement (i.e. progressive residual connections).
* Detection maps stay at 1/2 spatial res, aren't downscaled at 1/4 anymore.
* Learning rate peaks increase by 3% now
*Training from scratch: after 8h everything converged: attention loss got to 1.07 (last best was 1.06). Detection converged to 0.175 (last best was 0.2). Attention maps look OK, detection bad: all maps look the same!

Overall happy : the mid stem and attention got notably reduced and still performed. detection still has to be fixed.



##### ???
* The att/20 line helps at the beginning, but now that we are using pretrained, removing it worked out well, and **could help to use the capacity of the model better**. At this point may be late though (dead cells), so did the following:
  * Replaced att/20 by a function of the number of steps: att / (1 + 19*e^(-lambda*x)) which will smoothly go from 1/20 to 1/1. Training from scratch, found out that lambda=0.003 is stable, but lower may help regularize better (we go for 0.001). Maps also look crisper.

* In hope that the new 1/20 policy helps the attention module use its capacity better, we reduce the planes from 100 to 80.
* Also, we cat the alt image right after the stem, so the attention CAMs also read it directly.

* In a similar fashion but in the detector, isolated the cause for all heatmaps being equal: WITH the `det = det/20` all looks equal, removing it, lots of saturation (but also differences). Removed the line with the hope that SGD deals with the saturation.

Trying to fix detection without losing attention:
  * Set `background_factor` from 0.5 to 1 and DET_POS_WEIGHT from 7 to 100. Increased train stddev from 5 to 7.
  * TB detection logger now gets sigmoid instead of normalized maps, to match the optimizer.
  * The attention is used to multiply the cat(stem, alt) maps. Parallel to that, an MLP processes the alt img and generates also ``inplanes`` channels. Both are concatenated and passed to the detection CAMs.



IDEA NEXT: 

The detector seems to lack a lot of context. It focuses on very local feats. Use intermediate supervision instead of prog residual? Make a parallel hi-res stem that gets residuals from the other?

















## Rationale

The reasons are probably the target imbalances explained in the *SimplePose* paper, which heavily bias the model towards learning the background. Further reasons could be the student's lack of capacity or the inadequateness of the transfered stem.


## Open Items

* Input/breakpoint function with timeout that optionally jumps into debugger every x steps. This would allow for interventions during long training sessions.


```
from skimage.color import rgb2lab
input_features[2:5] = rgb2lab(img).transpose(2, 0, 1)
```

* pick higherhr and replace bottlenecks with step units. also add hard kp mining. also they add the loss of all scales with equal weights, is this ok? effect of feature concatenation: do we want this? definitely use dark plugin.

* Hourglasses are bad. Use step NNs or HRNet
* Train with focal loss? No, hard mining is better.
* Use squeeze-excitation modules? check the refining mechanisms, some of them have it already.

* Mix bottom-up with top-down? Are megvii using PAFs at all? IDEA: RSN paper shows that RSN outlines human bodies very well. So we can get rid of detector? Yes we can, see HigherHRNet.

* Use the refine-machine not only in the last stage? That is the idea of Progressive Context Refinement.

* Using image information to refine CRF-like and skip all the step and attention business? E.g. DARK goes in that direction, but maybe using CRFs gives an even better distribution?

* what is the role of 3D? What about using simulated data to reproject to 2D?

* How does the architecture in the AE paper look like for producing the embeddings? they dont tell... -> They simply output more channels at the NN end, and train them with a separate objective.


* 3D nice idea (from VNect?): During inference, change the bone lengths that are concatenated in the middle of the NN with the ones that we want to enforce, leaving the rest unchanged

* 3D: they propose using RNNs and such. Why not kalman filters?

* Optimization trick: after training, merge batch norm. with preceding weights to speedup inference. Also when going 3D we can distillate less joints and still group them via AEs.

2. If we can get a pretrained RSN, check runtime and evaluate using pycocoapi. Same with SimplePose.

* HRNet code: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

It seems that the HRNet_w32 with 7.1 GFLOPs and 0.75 COCO is a good candidate for distillation:
https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

* We have tried several students, latest PCR with dilations up to 12. no meaningful output came out.
* We added keypoint mining strategies. seem to have a good impact (more informative gradients?) but still no good outcome
* Selected easy minitrain datasets with small/med/big people on them. Still no good outcome
* Inspected pretrained stem outputs: they look very informative and meaningful.
* Went back to pose distillation paper for arch specifics: student models are 100MB? The teacher is 130MB. And HRNet student is same arch definition? that could be a reason, investigate further?


### Software TODO:

* Students: finish arch definitions (does having multiple relus matter?). Ideally we also have this inter-resolution pooling from HRNet. The relation between spatial res and depth should be capacity-conserving when needed, as described in the paper, "following the resnet principle of doubling depth while halving space". Ideally we want to end up with about 20GFLOPS at most. Finish training script and run it.

* Add hist of weights for stem and heatmap outputs to see improvement.

* Incorporate train fn to engine if abstract enough and add a call after every epoch. epochs should start at 1

This is our teacher mAP on 100 images (can be run in `dataloader_demo.py`):

```


```


* At some point we will want hard keypoint mining and DARK extensions.





* Our HHRNet inference script is bugged: Simplifications are expected to drop performance, but probably not by this much. Heatmaps look good, so issues are probably with grouping. This is the original HHRNet val5000 using our script (AP=0.41 instead of 0.69). Check the last step from `grouped` to `final_results` in their validation protocol. It could also be that the pooling NMS size should be bigger since we are upsampling:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.418
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.670
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.425
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.317
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.576
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.741
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.555
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.409
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.700
 #
 'AP', 0.41780704651640155,
 'Ap .5', 0.6701423661106006,
 'AP .75', 0.4251923088205309,
 'AP (M)', 0.31656746656877627,
 'AP (L)', 0.5764849426012213,
 'AR', 0.5312185138539042,
 'AR .5', 0.7407115869017632,
 'AR .75', 0.5546284634760705,
 'AR (M)', 0.4087680961485932,
 'AR (L)', 0.6997770345596432
```




## Role of Associative Embedding Dimensionality

The AE paper shows that 1 dimension is enough to achieve good performance. HigherHRNet uses 17 (one per joint), which is interesting since AE is seemingly unrelated to the number of joints. Also by qualitative observation it can be seen that all 17 dimensions look very similar. This posed the question, what would happen if we just pick one of them, or average them. The result is a noticeable drop in performance:


#### Original HHRNet COCO validation (AP=0.698):

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.698
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.872
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.761
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.655
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.764
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.741
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.891
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.792
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.688
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.817
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| pose_hig... | 0.698 | 0.872 | 0.761 | 0.655 | 0.764 | 0.741 | 0.891 | 0.792 | 0.688 | 0.817 |
```


##### Original HHRNet COCO validation with only 1st AE dimension (AP=0.621):

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.621
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.846
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.655
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.582
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.677
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.676
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.873
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.710
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.618
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.759
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| pose_hig... | 0.621 | 0.846 | 0.655 | 0.582 | 0.677 | 0.676 | 0.873 | 0.710 | 0.618 | 0.759 |
```


#### Original HHRNet COCO validation with all AE dimensions averaged (AP=0.593):

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.593
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.824
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.621
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.567
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.632
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.651
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.858
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.679
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.606
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.715
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| pose_hig... | 0.593 | 0.824 | 0.621 | 0.567 | 0.632 | 0.651 | 0.858 | 0.679 | 0.606 | 0.715 |
```


Still, we are practically halving the size of the NN output. Therefore this dimensionality reduction could bear room for improvement in terms of efficiency of the student and we settle for 1 dimension as in the original paper.



