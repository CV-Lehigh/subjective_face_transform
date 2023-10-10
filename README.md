# Subjective Face Transform

Paper link: https://arxiv.org/abs/2309.15381

Abstract: Humans tend to form quick subjective first impressions of non-physical attributes when seeing someone's face, such as perceived trustworthiness or attractiveness. To understand what variations in a face lead to different subjective impressions, this work uses generative models to find semantically meaningful edits to a face image that change perceived attributes. Unlike prior work that relied on statistical manipulation in feature space, our end-to-end framework considers trade-offs between preserving identity and changing perceptual attributes. It maps identity-preserving latent space directions to changes in attribute scores, enabling transformation of any input face along an attribute axis according to a target change. We train on real and synthetic faces, evaluate for in-domain and out-of-domain images using predictive models and human ratings, demonstrating the generalizability of our approach. Ultimately, such a framework can be used to understand and explain biases in subjective interpretation of faces that are not dependent on the identity.

![teaser](teaser.png)

## Download and set-up environment

git clone https://github.com/CV-Lehigh/subjective_face_transform.git

cd subjective_face_transform

conda env create -f environment.yml

## Dependencies

#### First Impression Prediction Models

Here is a list of our trained first impression prediction models, with a ResNet-18 base architecture. Empirically Chosen (EC) models for each attribute are as per results presented in the paper.

Attractiveness: AFLW, AFLW &#8594 (fine-tune) OMI

Dominance: AFLW, AFLW &#8594 (fine-tune) OMI

Trustworthiness: SCUT-FBP5500, SCUT-FBP5500 &#8594 (fine-tune) OMI
