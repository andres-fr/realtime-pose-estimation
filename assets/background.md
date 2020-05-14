# BACKGROUND:

This section summarizes the main ideas captured from a brief review of the recent literature (check the [BibTex references](references.bib). Surely a lot is left behind. There is agreement across the literature that for good human pose estimation both locally precise features as well as relatively broad contextual features are needed. While there is great variety in aproaches even within DL, all best performing models seem to address this directly in one way or another. The models are categorized in 2 practices:

* Top-Down: First, all person instances are detected. Then, for each person, the keypoints are found.
* Bottom-up: First, find all the separate human keypoints (and optionally limbs) in an image. Then assemble them into people.

Bottom-up models have the advantage of not depending on *a priori* human detectors and not having to run once per person. On the other hand they perform worse than top-down, since top-down methods can adjust the size of the input human crop, which makes the problem simpler in terms of scale (a crucial factor).

Here, some of the best performing approaches and their main ideas are outlined. Techniques to make them faster and 3D are also covered.

## Bottom-Up

### PersonLab:

* PersonLab paper: http://arxiv.org/abs/1803.08225

### PifPaf:

TODO

### SimplePose:

 SimplePose is a bottom-up method inspired by CMU OpenPose that **achieves 68.1 on COCO AP and runs over 30FPS on GPU, 5fps on CPU**.

* SimplePose paper: https://arxiv.org/abs/1911.10529
* OpenPose paper: https://arxiv.org/abs/1812.08008

Hourglass NNs are hierarchical nets that can be trained on different scales simultaneously. Particularly, SimplePose proposes a residual variant of hourglass-based NNs as a backbone for a multi-stage framework, introducing several effective ideas:

* Redundant elliptic heatmaps instead of PAFs for more effective encoding of connections
* Identity mappings and Squeeze-Excitation modules to enforce spatial and inter-channel attention
* Focal loss to correct imbalances between heatmap background/foreground and easy/hard keypoints.


### Associative Embeddings:

* Paper: https://arxiv.org/abs/1611.05424

This work tackles problems where detected instances have to be grouped, as in the case of multi-person, bottom-up human pose estimation. Unlike other common grouping approaches like spectral clustering, CRFs and generative probabilistic models, which usually rely on pre-computed or ad-hoc affinity measures and take place in 2-stage setups (detection then grouping), **associative embeddings are provided jointly with the detections, and work with few modifications on models, without requiring special design for grouping**. This is advantageous assuming that detection and association are highly correlated tasks.

* AEs can be used as an alternative to Part Affinity Fields (OpenPose) and elliptic heatmaps (SimplePose). In fact, under same model capacity it seems to improve COCO performance greatly over them. Unlike related techniques like spectral embeddings, here no specific spectral semantics have to be computed, the model is free to encode the assotiation semantics.

* Heatmap detections are trained as usual (2 loss against gaussian ground truth). Loss function for AEs has 2 terms: one minimizing the **tag variance within a person**, and one minimizing the **(pairwise) RBF affinity between every pair** of people. This enforces that embedded representations for separate people are both consistent and distinct.

* **1-Dimensional embeddings work well for this problem**. Adding dimensions didn't improve much. Also, when considering detection vs. grouping, replacing detections with ground truth (but keeping the tags) skyrockets performance, suggesting that **the bottleneck in performance is detection, not AE grouping**.


## Top-Down

### Megvii:

Megvii are currently leading the [COCO Leaderboard](http://cocodataset.org/#keypoints-leaderboard) with an **AP of almost 80 on `dev`**, by using their top-down Multi-Stage Pose framework, and making use of the RSN blocks. For the person detector, they use MegDet, but they conclude that any minimally good detector will work. Responsiveness in a real-time environment is to be tested; the papers provide computational size in GFLOPS.

* MSPN: https://arxiv.org/abs/1901.00148  
* RSN: https://arxiv.org/abs/2003.04030, http://arxiv.org/abs/2003.07232
* MegDet: https://arxiv.org/abs/1711.07240
* CPN (presenting GlobalNet): https://arxiv.org/abs/1711.07319

In the MSPN paper, they propose the usage of multi-stage also on top-down, introducing several effective ideas:

* Repeatedly down- and upsampling leads to loss of information especially on the downsampling step. Hourglass models are bad since they are symmetrical. Increasing channels on downsampling and retaining shallow channels on upsampling distributes the computational capacity much more effectively. For that they use their ResNet-based GlobalNet instead, but other backbones work.

* Cross-stage aggregation: At a given scale, down- and up-going features of a stage are aggregated into the down-going features of the next stage (at the same scale). This helps preventing information loss across stages.

* Coarse-to-fine: As the stages advance, the ground truth gaussian heatmaps will decrease their spread, to enforce better localization. Intra-stage refinement is also enforced via simultaneous multi-scale objective optimization.

* Online hard keypoint mining: If e.g. a person has 17 keypoints, the GlobalNet is trained on L2 loss for all of them. But the RefineNet only on the `M` worst-performing ones, thus being forced to focus on the hard ones. `M=8` worked best. SimplePose paper states that this is better than focal loss.


The following contributions are presented in the RSN paper and go on the top of MSPN:

* Intra-scale aggregation: We saw that simultaneous multi-scale training brings context and details together, but this is coarse. RSNs address this by also aggregating information within the same scale, leading to preciser localizations and great score improvement.

* This is inspired by DenseNets, but less dense, which translates into lower computational and memory demands and less redundant features. I.e. RSBs maintain efficiency and effectiveness in bigger architectures.

* An attention module on the top (Pose Refine Machine or PRM) further refines the predictions: Like SimplePose, it incorporates inter-channel attention via squeeze-excitation. It combines it with spatial attention to yield notable score gains efficiently.


### HRNet:

The HRNet group of papers successfully implement ideas from other works in both bottom-up and top-down fashions.

#### HRNet: http://arxiv.org/abs/1902.09212

This is a top-down setup that first runs a detector and then a single-stage NN for each person. The main idea here is that most other approaches go from high-to-low resolution to extract semantics, and then trace back from low-to-high to achieve localization (possibly fusing features along the way). E.g. both HourGlass and GlobalNet do so, with the difference that GN is not symmetrical (heavyweight downsampling and lightweight upsampling).

* Instead, HRNet *maintains* the high-resolution pipeline throughout the process, increasingly adding lower-resolution subnets and fusing activations along the way. This is different because **multi-scale fusion is performed without requiring intermediate supervision**. This translates into both accuracy and efficiency in computational/capacity demands.

* Another key idea is that multi-resolution representations are **fused repeatedly** along the inference. For that, a lightweight "exchange unit" is propsed, which resamples and adds the representations (strided conv for downsampling, nearest interp. for upsampling).

* Ablation study shows efectiveness of having multiple exchange units, as well as incrementally adding lower-resolution pipelines in a "stepwise" manner.

* Page 8 says that the network outputs heatmaps once per resolution. Page 4 says that they pick only the highest one, and empirically works well.

#### BytedanceHRnet: http://cocodataset.org/files/keypoints_2019_reports/ByteDanceHRNet.pdf

This is a follow-up by the same authors. They add a pre-trained apose refinement network (PoseFix) after the regular HRNet, and make a 6-element ensemble.

* PoseFix paper: https://arxiv.org/abs/1812.03595

Note that Progressive Context Refinement (PCR) can also be used as refiner with potentially better results.


## Hybrids/Plugins

### Higher HRNet: https://arxiv.org/abs/1908.10357

This paper presents several major deviations from HRNet:

* The HRNet backbone is converted to a **bottom-up** (BU) method. Unike other BU-methods, here multi-scale issues are explicitly tackled, with an emphasis on high (spatial) resolution feature maps. It achieves >70AP on COCO. Improvement is moderate on high-size person instances, and very high on mid-size.

* The resolution of the feature maps starts from 1/4, unlike prior BU models that start on 1/32. The strategy to retain/achieve HR is to 1) Use HRNet, 2) Use **deconvolution** on top of that. SimpleBaseline showed that deconv can be both efficient and effective in achieving that. Specifically, the output of HRNet is 1/4, and each deconv upscales by 2. The number of deconvs depends on the people size distribution. For COCO, they found 1 deconv step to be optimal. They claim that no post-processing is done, but after the deconv they add 4 residual blocks for refinement (Table 4 shows their contribution).

* It incorporates **multi-resolution supervision** by making one ground truth heatmap per scale, but all of them having the same standard deviation (2 pixels). The objective is the sum of the L2 losses for each respective scale. At inference, they upsample (bilinear interp.) each predicted heatmap to image size, and average.

* Grouping of extracted heatmaps is done via adding **associative embeddings**.

### DarkPose: http://cocodataset.org/files/keypoints_2019_reports/DarkPose.pdf

Keypoints are usually encoded as biased distributions (e.g. gaussian) and decoded rudimentarily (e.g. NMS). Little attention has been paid to this although precise localization is crucial. This work aims to address this with a Distribution-Aware coordinate Representation of Keypoints (DARK).

* Encoding: Also using gaussian heatmaps, but **the mean is not quantized**.

* Decoding: Taylor-based method to estimate the sub-pixel peak of the prediction.

* The report presents a top-bottom setup, but this method is **independent of the architecture**. Can be used as a plug-in to notably boost performance, and is especially helpful to mitigate the quality loss when reducing input size, which is helpful for **faster inference**.

### Progressive Context Refinement: https://arxiv.org/pdf/1910.12223.pdf

A top-down method is presented, with several contributions and very good performance on COCO:

* The **Context-Aware Module** (CAM) is a block that directly produces heatmaps, and features 3 pipelines: residual, squeeze-excite and dilated convolutional. It is aimed to (residually) capture the context while having the channel-wise attention provided by SE. The PCR module consists of a stack of CAMs, where the output of a CAM is being added to all the prior ones before passed to the loss function. This way, each CAM **refines** the HMs of the prior ones. The CAM can be appended to different backbones, e.g. HRNet.

* Training strategies: since the setup is top-down, Hard-Negative Person Detection Mining (HNDM) helps overcome false positives by the detector, by setting the ground-truth heatmaps to zero. This way the refiner learns to ignore the false positive detections confidently. They also propose a semi-supervised method using unlabeled COCO data and annotations from the AIC dataset.



## Applications:


### Fast Pose Distillation: http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Fast_Human_Pose_Estimation_CVPR_2019_paper.pdf

This work shows that traditional 2-stage distillation is extremely effective for this problem, **reducing model size by a factor of 6 while keeping 95% generalization ability**. Evaluations are provided on MPII and **not** on COCO.

* The 2-stage distillation process consists in first training a heavyweight teacher model and then using its predictions to train a lightweight student model. Only the heatmaps are used, so any teachers/students that outputs them will be suitable.

* The proposed loss function is `alpha*A + (1-alpha)*B`, where A is the MSE loss between ground truth and student predictions, and B is the MSE loss between teacher and student. A wide range of `alpha` values work, 0.5 being a good one. XE was also tested but MSE seemed to work better for the experiments.

### VNect: https://arxiv.org/abs/1705.01583

This work targets **single-person, real-time 3D pose estimation from monocular images** (so implicitly it is top-down). For that, they propose a specialized CNN that jointly predicts 2D and 3D positions, together with a model-based kinematic skeleton fitting to produce temporally stable predictions. 3D predictions are given as offsets relative to the root (hips), with respect to a height-normalized skeleton (knee-neck:  92cm).

* The CNN is a modified ResNet-50. It predicts 2D heatmaps and 3D root-relative locations **jointly**. For that, the parent-relative 3D delta fields, and their L2 norm are also explicitly computed and supervised in the pipeline, to make the NN aware of bone lengths.

* The model is first trained on 2D, and then expanded to 3D, always in supervised manner. Check the abundant details about architecture and training procedure, as well as hacks and assumptions on section 4.1.

* For inference, the 2D heatmaps are temporally smoothed, and the 3D relative locations retrieved. The final 3D prediction is retrieved by minimizing a 4-part energy function that features an inverse kinematics element (weight=1, checks for angle similarities), a reprojected term (weight=44, projects back to 2D **needing knowledge on camera calibration** and compares with heatmaps), a smooth term (w=0.07, for temporal smoothness) and a depth term (w=0.11, penalizes large variations in predicted depth). The impact of these terms is analyzed at the end of 5.2.

* The model has the advantages of being fast and temporally+locally stable. It is sensitive to fast movements, occlusions, bounding box tracker, calibration and other assumptions. See sections 7 and 8 for more details.


### Lightweight 3D Human Pose Estimation Network Training Using Teacher-Student Learning: https://arxiv.org/abs/2001.05097

This pre-production work applies pose distillation to VNect (together with other techniques) to achieve mobile/embedded real-time 3D pose estimation. Accuracy is reported on Human3.6M, the largest 3D dataset with 11 subjects performing 15 actionsm via Mean Per Joint Position Error (MPJPE). **They decrease no. of parameters to 7% keeping performance at 83%**.

* Distillation paper: https://arxiv.org/abs/1503.02531

* The distilled student NN is found via manual model search, based on VNect. The bone lenghts are given in L1 instead of L2 to speed up. Also, only a subset of 15 joints are distilled.

* The heatmap loss function is as proposed in Fast Pose Distillation. The 3D loss is analogous, but replacing the heatmap L2 norm with the VNect 3D loss (i.e. the 3D loss is masked with the heatmaps).

* They are able to **restore the global position** from the bounding box using an equation presented in 3.3.

* The model doesn't know about anatomical restrictions. For this, the joint angles are regressed via inverse kinematics, and **joint rotation limits are applied as post-processing**.
