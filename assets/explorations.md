# EXPLORATIONS


## Rationale

The reasons are probably the target imbalances explained in the *SimplePose* paper, which heavily bias the model towards learning the background. Further reasons could be the student's lack of capacity or the inadequateness of the transfered stem. These are currently being explored:

  * Init with kaiming uniform may help? What about multiplying the heatmaps by a constant to increase their importance?


## Open Items

* Input/breakpoint function with timeout that optionally jumps into debugger every x steps. This would allow for interventions during long training sessions.

* second distillation stage: first train the student body to map from stem output to augmented teacher/GT. Then replace with a smaller, non-fp16 stem that learns to map from HSV/LAB to stem output.

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



