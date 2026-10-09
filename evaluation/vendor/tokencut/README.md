# TokenCut

`object_discovery.py`, `LICENSE`, and the fixed COCO20K filename list are
copied without changes from the authors' repository at commit
`fed52cd5b60891baefd8ec7110dafa73be816ee1`:
https://github.com/YangtaoWANG95/TokenCut

The evaluator uses its `ncut` function with final normalized patch **outputs**,
rather than attention keys, following DINOv3 Sec. 6.1.4 and Appendix D.4:
https://arxiv.org/html/2508.10104v1#A4.SS4

DINOv3's public repository at `6876159a11b4df116f30f667f8c9888617df0751`
contains no object-discovery evaluator. Consequently this reproduces its
published protocol using official TokenCut, rather than unpublished Meta code.
Full native resolution is preserved; normalized images are padded with zeros
on the right/bottom to the next patch multiple as in official TokenCut.
The paper does not specify padding, final LayerNorm, crowd handling, or an
IoU equality convention explicitly. Final normalized outputs are the standard
backbone outputs; dataset filtering/coordinates follow official TokenCut.
The primary metric uses IoU > 0.5 as described by DINOv3; IoU >= 0.5 from
the TokenCut release is saved as a secondary metric.
