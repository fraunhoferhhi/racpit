---
title: Unsupervised Domain Adaptation across FMCW Radar Configurations Using Margin Disparity Discrepancy
description: Supplementary material for EUSIPCO2022
---


**WORK UNDER CONSTRUCTION**

Placeholder for the additional material of the poster "Unsupervised Domain Adaptation
across FMCW Radar Configurations Using Margin Disparity Discrepancy" at EUSIPCO 2022.

# Presentation

Hi, my name is Rodrigo Hernangómez. I am a research associate with the
Fraunhofer Heinrich Hertz Institute for Telecommunications (HHI for
short), where I am also pursuing a PhD together with
the Technical University of Berlin.

This video is a presentation of our paper Unsupervised Domain Adaptation
across FMCW Radar Configurations Using Margin Disparity Discrepancy, as
part of the conference program at EUSIPCO 2022 in Belgrade.

## Cover

The first thing to say about this work is that it is the result of a
collaboration of HHI researchers, namely Igor Bjelakovic, Slawomir
Stanczak and myself, with Infineon Technologies (represented by Lorenzo
Servadei).

## Outline

Let us take a look at the outline. After a brief introduction to motivate the connection of domain adaptation and radar, I will dive deeper into both topics to highlight the theoretical and practical contributions of this work, respectively. These are supported by the results of our experiments, which will finally allow us to wrap it up with a conclusion.

## Motivation

HHI and Infineon have been collaborating on radar sensing for the past
few years. This research topic owes its growing interest to several
factors.

On the one hand, semiconductor companies have managed to produce highly
integrated radar chipsets thanks to the Frequency-modulated continuous
wave technology. An example of this being the 60 GHz sensor frontend
developed by Infineon that lies at the core of Google's Soli Project.

On the other hand, radar enables some interesting commercial
applications, such as human activity and surveillance or hand gesture
recognition as in the animation from the Soli project below.

On the other hand, radar sensors present some advantages to cameras when
it comes to privacy concerns and ill-posed optical scenarios, including
through-wall and bad-lighting situations.

However, same as in the realm /rɛlm/ of computer vision, the complexity
of the classification of radar signatures has driven radar approaches to
resort to machine learning and deep learning techniques that require
high amounts of data.

## DA intro

Deep learning's need for data poses a challenge to radar, especially
when aspects such as the sensor's configuration, the environment or user
settings have a large diversity.

Indeed, each of these may lead to changes in the data domain, which the
machine learning literature understands as the union of an input or
feature space X, an output or label space Y and a probability
distribution p over both spaces.

In particular, domain adaptation techniques consider the case where X
and Y remain equal but the underlying probability distribution differs
between a source domain S, from which we can draw sufficient training
data, and a target domain T that we ultimately aim for deployment.

Such a framework allows an easier deployment from pre-trained models
without caring too much about the aforementioned problems. Such a
pre-trained model can be agnostic to the radar's configuration, and only
require few labeled data from the target domain to fine-tune its model.
This is the case of the so-called supervised techniques, which we
precisely investigated in this reference. In the current work, we have
moved on to the more demanding situation where the target data is
provided without labels from the label space Y, also called unsupervised
domain adaptation.

## MDD

More specifically, we have turned to an unsupervised domain adaptation
method known as margin disparity discrepancy, which has been recently
developed by Zhang et al. based on theoretical guarantees and tested on
computer vision datasets. Zhang et al. proved that the true error in the
target domain can be bounded by the empirical error in the source domain
plus a residual ideal loss and the so-called margin disparity
discrepancy (MDD).

In order to understand MDD, we first must define the margin disparity as
an unsupervised metric that compares two arbitrary classifiers f' and f
without labels, only as the loss of f' with respect to the predictions
by f for certain features x.

MDD is then calculated as the maximum difference on margin disparity
between the source and target domains for all possible f', and by doing
so, it behaves as a measure of domain shift. Since we want to minimize
MDD to obtain a lower bound on the target error, MDD becomes a min-max
problem. This fact will become relevant in the practical implementation.

## Cross-entropy bound

However, let us dwell on theory for one more slide. The authors of MDD
derived their bounds using the ramp loss, which you can spot as the red
dashed line for this uni-dimensional example. While this loss has some
nice theoretical properties, its non-convexity renders it unpractical.
For this reason, the cross-entropy loss (in purple) is used for
experiments in the MDD paper.

On top of the valuable theoretical remarks on the said paper why the
choice of the cross-entropy loss is sensible, we have managed to upper
bound the empirical source error in the previous slide, which originally
uses the ramp loss, by its cross-entropy counterpart.

We will skip the proof of this bound for brevity, but you can check it
in the paper if you are interested. A second look at the plot though,
allows us to grasp the intuition behind it. For this simple
unidimensional case, the cross-entropy can be regarded as a smooth
version of the green-dotted hinge loss, which in turn is nothing more
than a convex relaxation of the ramp loss.

Another noteworthy detail is the inclusion of the soft-margin softmax
from Liang et al. in the cross-entropy, which is actually crucial for
the bound derivation. Again, all of this is included in the paper.

# Radar data

## Measurement campaign

Back to the radar discussion, we should refer to the benchmark we have
used for our domain adaptation experiments.

Here we have chosen the use case of human activity classification that
we have been working on for four years, but the principles are also
valid for other applications. We consider 5 human activities:

-   Standing in front of Infineon's sensor.

-   Waving with the hand at the sensor.

-   Walking back and forth to the sensor.

-   Boxing while standing still.

-   And Boxing while walking towards the sensor, which can be seen in
    the picture below.

There you can also note the position of four BGT60 sensors by Infineon
that were set up at their data acquisition lab in Dresden, Germany. A
detail of this commercial 60 GHz FMCW sensor is visible as well. Each of
the four sensors captured the same activities at the same time with a
different radar configuration. Thus, important settings of FMCW radar
such as the number of samples per chirp, the bandwidth or the frame
period varies across configurations, which in turn affects the
resolution and scope that can be achieved by the radar in range and
speed. We consider each separate FMCW configuration as a different
domain.

## Implementation

Range and speed (also known as Doppler shift) of FMCW radar targets can
be explored through so-called Range-Doppler maps. These reflect the
returned energy off the targets over both dimensions and they are
obtained frame by frame through a two-dimensional Fourier transform from
the raw returned signal across samples and chirps, respectively.
Furthermore, we resize these series of maps accordingly so that the
feature space x always presents the same dimensions across radar
configurations. The resolution and noise characteristics remain specific
for each configuration, though, and so does the probability distribution
of each domain.

Moreover, we stack these Range Doppler maps across frames and sum over
the corresponding axis to obtain range and Doppler spectrograms. *(start
video)* That is, intensity plots of time vs. distance and speed,
respectively, where we can identify the target walk as the main
reflected target, but also the micro-doppler signatures of the fists
while boxing, which orbitate around the main position of the body.

This allows us to train a light neural network that takes both range and
Doppler dynamic signatures and extracts the most relevant features from
them through two separate branches of 3 convolutional layers, shown in
red and green. Then, the extracted features undergo their path through
the f and f' classifiers according to the same implementation as in
MDD's original paper. That is, f and f' are two adversarial fully
connected neural networks, depicted here in white and blue,
respectively.

In order to solve the min-max problem that I mentioned before, we make
use of a gradient reversal layer (marked as GRL on the diagram) to
maximize the mdd loss term on $f'$ while minimizing on the convolutional
layers.

This is actually the practical implementation of the min-max problem
that I mentioned before, as suggested by the original authors of MDD.

# Evaluation

## Results

Our MDD implementation delivers the following results for all possible
combinations of source and target configuration in our radar dataset.

As you can see, the classification accuracy exceeds 85% in all cases.
This is comparable to the results in gray from our previous work, where
we used the supervised domain adaptation method of FADA, or Few-shot
Adversarial Domain Adaptation.

Furthermore, the results of MDD for radar are consistent with the
results reported by the original authors for different computer vision
datasets. As such, both the average accuracy of MDD for the radar and
office-31 datasets lies around 89%. For the more challenging visual
datasets of Office-Home and VisDA, MDD performs better with the radar
datasets for all domain pairs.

We also tried to investigate the performance of MDD when replacing the
original soft-max function with the soft-margin version proposed by
Liang et al. but we could not find significant differences for any
dataset.

## Conclusion

In conclusion, we have confirmed that MDD as proposed by Zhang et al.
works with our radar dataset as well as it does for computer vision
data.

If we focus exclusively on our work on radar, we can say that this
unsupervised technique shows comparable performance with the supervised
methods we tried in the past, while it has the benefit of working with
unlabeled data, thus circumventing the burden of labelling measurements.

As a last comment on the soft-margin softmax, we cannot claim any
practical impact with our results at hand. Nevertheless, we must say
that soft-margin softmax is vital our theoretical contribution in this
work to derive a bound with the cross-entropy.


# References

1.  Hernangómez, R., Santra, A. and Stańczak, S. (2019) ‘Human Activity Classification with Frequency Modulated Continuous Wave Radar Using Deep Convolutional Neural Networks’, in _2019 International Radar Conference (RADAR)_. _2019 International Radar Conference (RADAR)_, Toulon, France: IEEE, pp. 1–6. Available at: https://doi.org/10.1109/RADAR41533.2019.171243.
2.  Hernangómez, R., Santra, A. and Stańczak, S. (2021) ‘Study on feature processing schemes for deep-learning-based human activity classification using frequency-modulated continuous-wave radar’, _IET Radar, Sonar & Navigation_, 15(8), pp. 932–944. Available at: https://doi.org/10.1049/rsn2.12066.
3.  Khodabakhshandeh, H., Visentin, T., Hernangómez, R. and Pütz, M. (2021) ‘Domain Adaptation Across Configurations of FMCW Radar for Deep Learning Based Human Activity Classification’, in _2021 21st International Radar Symposium (IRS)_. _2021 21st International Radar Symposium (IRS)_, Berlin, Germany, pp. 1–10. Available at: https://doi.org/10.23919/IRS51887.2021.9466179.
4.  Liang, X., Wang, X., Lei, Z., Liao, S. and Li, S.Z. (2017) ‘Soft-Margin Softmax for Deep Classification’, in D. Liu, S. Xie, Y. Li, D. Zhao, and E.-S.M. El-Alfy (eds) _Neural Information Processing_. Cham: Springer International Publishing (Lecture Notes in Computer Science), pp. 413–421. Available at: https://doi.org/10.1007/978-3-319-70096-0_43.
5.  Lien, J., Gillian, N., Karagozler, M.E., Amihood, P., Schwesig, C., Olson, E., Raja, H. and Poupyrev, I. (2016) ‘Soli: ubiquitous gesture sensing with millimeter wave radar’, _ACM Transactions on Graphics_, 35(4), p. 142:1-142:19. Available at: https://doi.org/10.1145/2897824.2925953.
6.  Motiian, S., Jones, Q., Iranmanesh, S. and Doretto, G. (2017) ‘Few-Shot Adversarial Domain Adaptation’, _Advances in Neural Information Processing Systems_, 30\. Available at: https://proceedings.neurips.cc/paper/2017/hash/21c5bba1dd6aed9ab48c2b34c1a0adde-Abstract.html.
7.  Santra, A. and Hazra, S. (2020) _Deep learning applications of short-range radars_. Artech House.
8.  Zhang, Y., Liu, T., Long, M. and Jordan, M. (2019) ‘Bridging Theory and Algorithm for Domain Adaptation’, in _International Conference on Machine Learning_. _International Conference on Machine Learning_, PMLR, pp. 7404–7413. Available at: http://proceedings.mlr.press/v97/zhang19i.html.
