# NeCo comparisons: original iBOT versus Region ViT-S/16 epoch 200

The implementations in `vendor/neco` and `vendor/hummingbird` are copied from
the authors' repositories, with MIT licenses and commits recorded in
`vendor/neco_provenance.json`. Evaluation mathematics and transforms are not
rewritten. `neco_benchmarks.py` adapts checkpoint loading and local logging.
Three package markers isolate the upstream `data`, `src` and `experiments`
namespaces from similarly named packages in this repository.

Sources:

- [NeCo paper, Appendix A.2](https://proceedings.iclr.cc/paper_files/paper/2025/file/fbc9981dd6316378aee7fd5975250f21-Paper-Conference.pdf)
- [NeCo released evaluation code](https://github.com/vpariza/NeCo)
- [Contemporary Open Hummingbird v1.x code and subset lists](https://github.com/vpariza/open-hummingbird-eval/tree/v1.x)

## Features and comparison

Both checkpoints load the teacher backbone strictly, excluding iBOT projection
heads. The released NeCo backbone returns final-block patch tokens after final
LayerNorm. No softmax or last-four-block averaging is introduced. Every backbone
parameter is frozen. A preflight compares NeCo outputs with our usual backbone.
Each run stores its resolved checkpoint, upstream commit, configuration and
metrics. Original iBOT is the uncontinued pretrained checkpoint; Region is the
continuation-200 checkpoint that scored 64.28 VOC online k-NN mIoU.

## Dense retrieval / in-context segmentation

VOC2012 and ADE20K; full support set and fractions 1/8, 1/64, 1/128. Each partial
fraction uses the five released subset lists, not newly sampled splits. Full
support uses one seed. Resolution 512 for patch size 16 follows the authors'
Hummingbird reproduction. Memory capacity 10,240,000; ScaNN; 30 neighbours;
upstream patch-label histograms, class-balanced patch sampler, image transforms,
feature normalization, temperature and mIoU implementation.

The publication does not specify how many augmented passes populate each small
support set. We use at least two, increasing this to the minimum number needed
to populate that same capacity without requesting more patches per image than
exist. This choice is recorded per task and applied identically to both models;
an exact reproduction of an undisclosed augmentation schedule cannot be claimed.

Two bookkeeping fixes are explicit: keep incomplete VOC batches so every support
and validation image is used; trim the unpopulated remainder of the preallocated
bank before building ScaNN. Neither changes the released matching algorithm.
The released blue-channel normalization std of 0.255 is preserved in this task.

## Unsupervised clustering

VOC2012, ADE20K, COCO-Things and COCO-Stuff: K equal to the semantic class count,
and K=500 overclustering. Each task runs all five clustering seeds in the released
code. Input 448; masks 100; StandardScaler and PCA to 50 dimensions;
FAISS K-means; released Hungarian/many-to-one mappings and metric. This is the
released global clustering evaluation, not the separate CBFE/community detection
pipeline. The upstream VOC clustering loader drops the incomplete validation
batch (1440 of 1449 images at batch 32); this behaviour is retained here.

`evaluation_neco_k300_cbfe.yaml` adds K=300 for all four datasets using the same
five-seed released evaluator, without repeating the existing K=GT/K=500 jobs.

## Fully unsupervised semantic segmentation: CBFE + community detection

`neco_fully_unsupervised.py` calls the pinned released `start_unsup_seg` on VOC2012
train (1464) and the entire validation set (1449). Input 448, masks 100, final-LN
teacher patch features, StandardScaler/PCA50. The initial foreground clustering
uses spherical K-means with K=200, followed by foreground-only K=20 (five seeds)
for CBFE segmentation and K=189 for community detection. Both checkpoints use
the paper's fixed weight threshold 0.07 and Markov time 1.2. Ten Infomap runs are
saved individually, with mean/std and best reported separately. No independent
model-specific hyperparameter sweep is performed.

The paper describes 70% attention mass, while the released helper uses 65%; this
evaluation uses the paper's 70%. All other attention processing is unchanged.
Runtime compatibility patches keep the vendor source untouched: alias renamed
`greycomatrix`; retain fractional co-occurrences in a float64 accumulator
(upstream integer in-place addition raises with current PyTorch); copy generated
attention masks to the path CBFE reads; use valid Infomap seeds 1--10. Each model
has its own cache and masks, on scratch storage.

The released code uses **training foreground ground truth to select the CBFE
foreground threshold**. It fits the scaler/PCA/clusters and computes the graph
on combined train and validation features; those transductive choices are
preserved and recorded. The "fully unsupervised" benchmark name follows NeCo;
it should not be described as having no annotation-dependent calibration.
Because of these released-code repairs and paper/code differences, this is a
documented NeCo implementation comparison, rather than a byte-for-byte rerun of
the unpublished experiment that generated the paper's score.

## COCO-Things and COCO-Stuff linear segmentation

Official COCO2017 images, panoptic semantic category masks for Things, and
COCO-Stuff2017 masks for Stuff. The unchanged loader maps labels to 12 Things
and 15 Stuff supercategories. Seed 400; upstream 10% training selection;
all 5000 validation images. Input 448, batch 128, frozen backbone plus linear
1x1 convolution; SGD lr 0.01, momentum 0.9, weight decay 0.0001; StepLR at 20.

The paper specifies 20 epochs, whereas released YAML configs specify 25. These
runs use the paper's 20 epochs, with the difference recorded. Training/selection
uses the released 100-pixel masks; the selected head also runs through the
authors' final `eval_linear.py` evaluator with its default 448-pixel masks.
Both scores are retained with their resolutions, rather than conflated.

## Semantic region retrieval

`semantic_region_retrieval.py` is a separate custom, mask-conditioned analysis.
VOC2012 train (1464 images) is the gallery, validation (1449) the query set.
For each non-background semantic class in an image, area-weighted mask pooling
of final-LayerNorm patch tokens produces a region descriptor, then L2-normalized.
Input 448. Similarity is cosine. Report per-class and macro/micro mAP and
Recall@1/5/10; no split overlap, fitted scaler, tuning or augmentation.
Ground-truth masks define the regions, so this is representation retrieval,
not automatic region localization or a published NeCo benchmark.
