# Photonics Optimization Testbed

This repository contains source code and data for the paper:

* Mo Chen, Rasmus E. Christiansen, Jonathan A. Fan, Göktuğ Işiklar, Jiaqi Jiang, Steven G. Johnson, Wenchao Ma, Owen D. Miller, Ardavan Oskooi, Martin F. Schubert, Fengwen Wang, Ian A. D. Williamson, Wenjin Xue, and You Zhou, "Validation and characterization of algorithms and software for photonics inverse design", *J. Opt. Soc. Am. B*, in press (2024).

In this work, we present a reproducible suite of test problems for large-scale optimization ("inverse design" and "topology optimization") in photonics,
where the prevalence of irregular, non-intuitive geometries can otherwise make it challenging to be confident that new algorithms and software are functioning as claimed. 
We include test problems that exercise a wide array of physical and mathematical features — far-field metalenses, 2d and 3d mode converters, resonant emission and focusing,
and dispersion/eigenvalue engineering — and introduce an *a posteriori* lengthscale metric for comparing designs produced by disparate algorithms.
For each problem, we incorporate cross-checks against multiple independent software packages and algorithms, and reproducible designs and their validations scripts
are included as supplementary information. We believe that this suite should make it much easier to develop, validate, and gain trust in future inverse-design approaches and software.

Each subdirectory of this repository corresponds to one of the test problems described in our paper.  See the corresponding README files for short descriptions, and the paper for more details.
