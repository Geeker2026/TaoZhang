# Third-party notices

## Entropy Rate Superpixel Segmentation (ERS)

Copyright 2011, Ming-Yu Liu. Authors: Ming-Yu Liu, Oncel Tuzel, Srikumar Ramalingam and Rama Chellappa. Reference: *Entropy Rate Superpixel Segmentation*, CVPR, 2011.

The original sources and notices are retained unchanged in `third_party/ers/source/` and `third_party/ers/UPSTREAM_README.txt`. ERS permits use, copying, modification and distribution for **non-commercial purposes**, subject to retaining the copyright and permission notices. The complete original terms in those files govern this dependency; this package does not relicense it.

`mex_ers.mexw64` is the Windows binary used for the manuscript experiments. Its SHA-256 (`2a88c19410fa6619730809158b1f00722df9aaa836f1eb51d5a67973e7f8070e`) matches the value recorded in the frozen segmentation metadata of those runs, so it is the file those runs used. Whether it was compiled from the exact source revision bundled here has not been independently verified; rebuilding from `source/` on another platform can produce a slightly different segmentation. `main.m` runs that build only when no MEX for the running platform is present.

## PCA helpers

`third_party/legacy_pca/frozen_pca_basis.m` and `frozen_pca_svd.m` retain Deng Cai's original PCA/mySVD author and version comments. Function/helper names were changed in the larger project to avoid path collisions; this minimal package copies those helpers unchanged. No additional license grant is assigned to these helpers here.

## Authors' MATLAB code

Released under the MIT license in `LICENSE`. That license covers only files authored by the paper's authors. Files under `third_party/` keep their own terms, and because this package redistributes ERS, the package as a whole is limited to non-commercial use.

## Figures

`docs/figures/fig1_tdlpp_framework.png` and `docs/figures/fig3_sppr_framework.png` are Figs. 1 and 3 of the published article (Pattern Recognition 184 (2027) 115271, © 2026 Elsevier), reproduced for documentation of the method. They are not covered by the MIT license. `docs/figures/pipeline.svg` is drawn for this release and follows the same license as the authors' code.
