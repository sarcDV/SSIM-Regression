# Reference-less SSIM Regression for Detection and Quantification of Motion Artefacts in Brain MRIs

Motion artefacts in magnetic resonance images can critically affect diagnosis and the quantification of image degradation due to their presence is required. Usually, image quality assessment is carried out by experts such as radiographers, radiologists and researchers.
However, subjective evaluation requires time and is strongly dependent on the experience
of the rater. In this work, an automated image quality assessment based on the structural
similarity index regression through ResNet models is presented. The results show that the
trained models are able to regress the SSIM values with high level of accuracy. When the
predicted SSIM values were grouped into 10 classes and compared against the ground-truth
motion classes, the best weighted accuracy of 89 ± 2% was observed with RN-18 model,
trained with contrast augmentation.

This work was presented at "Medical Imaging with Deep Learning, Zürich, 6 – 8 July 2022"
[MIDL 2022 SHORT PAPER PDF](https://openreview.net/pdf?id=24cqMfboXhH)
