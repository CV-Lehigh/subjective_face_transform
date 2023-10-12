# Subjective Face Transform

Paper link: https://arxiv.org/abs/2309.15381

Abstract: Humans tend to form quick subjective first impressions of non-physical attributes when seeing someone's face, such as perceived trustworthiness or attractiveness. To understand what variations in a face lead to different subjective impressions, this work uses generative models to find semantically meaningful edits to a face image that change perceived attributes. Unlike prior work that relied on statistical manipulation in feature space, our end-to-end framework considers trade-offs between preserving identity and changing perceptual attributes. It maps identity-preserving latent space directions to changes in attribute scores, enabling transformation of any input face along an attribute axis according to a target change. We train on real and synthetic faces, evaluate for in-domain and out-of-domain images using predictive models and human ratings, demonstrating the generalizability of our approach. Ultimately, such a framework can be used to understand and explain biases in subjective interpretation of faces that are not dependent on the identity.

![teaser](teaser.png)

## Download and set-up environment

```
git clone https://github.com/CV-Lehigh/subjective_face_transform.git

cd subjective_face_transform

conda env create -f environment.yml
```

## Dependencies

Create a folder *pretrained_dependencies/* to add all pretrained model dependencies.

#### Inversion Dependencies

Here is a list of all pretrained models required to *project and invert* face images using the inversion setup. Please download and add all relevant models to the path *pretrained_dependencies/image_inversion_models*.

Face landmarks model: [shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)

Pretrained HyperInverter checkpoint: [hyper_inverter_e4e_ffhq_encode_large.pt](https://drive.google.com/file/d/1JxKAHk-u4joVq1NmDsVcR_ov-cNWFBSu/view)

Pretrained StyleGAN2-ADA: [stylegan2-ada-ffhq.pkl](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) {Please download and rename to *stylegan2-ada-ffhq.pkl*}

w-space encoder: [w_encoder_e4e_ffhq_encode.pt](https://drive.google.com/file/d/1uVqnXDBujAv4a4TU99SFIwKeAG-H6pzp/view)

irse50 encoder checkpoint: [model_ir_se50.pth](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view)

#### First Impression Prediction Models

Here is a list of our trained first impression prediction models, with a ResNet-18 base architecture. Training datasets for each model are presented. Empirically Chosen {EC} model for each attribute is as per results shown in the paper. Please download and add all relevant models to the path *pretrained_dependencies/subjective_models*.

Attractiveness: [SCUT-FBP5500](https://drive.google.com/file/d/1EDXEzizBVljGYKohMyve-ILEgkfdh3lw/view?usp=sharing), [SCUT-FBP5500 &#8594; (fine-tune) OMI {EC}](https://drive.google.com/file/d/1b8zXDyX5hpblq5HQ4VB4dEj6DwSQha3L/view?usp=sharing)

Dominance: [AFLW](https://drive.google.com/file/d/1rDiNTEZ0PmHaCBFUzPYy8wOD-J59DLrC/view?usp=sharing), [AFLW &#8594; (fine-tune) OMI {EC}](https://drive.google.com/file/d/1nN1JKouM0fe7NzQ5Xgv0fx0mUBFd05TJ/view?usp=sharing)

Trustworthiness: [AFLW](https://drive.google.com/file/d/1FomGr7TGs8EZ9Bo5TZkyKpfbNHfjPqRQ/view?usp=sharing), [AFLW &#8594; (fine-tune) OMI {EC}](https://drive.google.com/file/d/1Gvyrehb_nDmMiPh9CmL2_PqZ1EdFqpeb/view?usp=sharing)

#### Mapping Models

Here is a list of our trained image⟷score mapping models, trained using the proposed pipeline. Training datasets for each model are presented. Empirically Chosen {EC} model for each attribute is as per results shown in the paper. Please download and add all relevant models to the path *pretrained_dependencies/mapping_models*.

Attractiveness: [FFHQ](https://drive.google.com/file/d/1Hrq5Ub9BVRbmAXaN1fSKmjpTv90WaI_j/view?usp=sharing), [StyleGAN2-ADA generated](https://drive.google.com/file/d/1mMbKZUbrxNYWOxcxmh476-NVu41-ybKr/view?usp=sharing), [CelebAMask-HQ {EC}](https://drive.google.com/file/d/1CcUs5jP9PYvbnmxG-vLYD_G3b0JWm7x5/view?usp=sharing)

Dominance: [FFHQ](https://drive.google.com/file/d/1xKXEHpeedemqyfFJF59J8oMj39W6Jh8G/view?usp=sharing), [StyleGAN2-ADA generated](https://drive.google.com/file/d/1lqT6YBxaK_hFTY7O8MVxTRblyZzKry3-/view?usp=sharing), [CelebAMask-HQ {EC}](https://drive.google.com/file/d/1VXec2WyleoV12M4MDkoEFoFLwbtXpSoU/view?usp=sharing)

Trustworthiness: [FFHQ](https://drive.google.com/file/d/1jlWOuB5M4kLSk8LCAHaCMRui9PKwEk65/view?usp=sharing), [StyleGAN2-ADA generated](https://drive.google.com/file/d/1OdZGlMzFWIyGCEjLWzwfBJjIF2iN5npo/view?usp=sharing), [CelebAMask-HQ {EC}](https://drive.google.com/file/d/1WL-zBNBPEaJuvKwxjk72N4ktNQ4F05Nb/view?usp=sharing)

## Generating Face Variations

Run the following file to generate variations of input set of face images along the score range of *{-0.2,0.2}*, with generation of corresponding attribute predictions.

```
python generate_image_variations.py
```

## License

We license our code under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). The code is released for academic research use only.

## Acknowledgements

We implement the [HyperInverter](https://github.com/VinAIResearch/HyperInverter) setup for [projection and inversion](https://github.com/CV-Lehigh/subjective_face_transform/tree/main#inversion-dependencies) of face images in the GAN latent space. The [first impression prediction](https://github.com/CV-Lehigh/subjective_face_transform/tree/main#first-impression-prediction-models) models are trained based on the setup and details described in the paper [Convolutional Neural Networks for Subjective Face Attributes](https://github.com/mel-2445/Predicting-First-Impressions). Our work builds upon the initial [StyleFlow](https://github.com/RameenAbdal/StyleFlow) implementation to create a [mapping model](https://github.com/CV-Lehigh/subjective_face_transform/tree/main#mapping-models) between the input faces and the corresponding subjective scores of attributes like Attractiveness, Dominance, and Trustworthiness.

We also thank all of the contributions that came prior to the above mentioned work. Notable mentions are [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), [e4e](https://github.com/omertov/encoder4editing), and [torchdiffeq](https://github.com/rtqichen/torchdiffeq).

## Citation

If helpful, please consider citing us as follows:

```
@article{roygaga2023subjective,
  title={Subjective Face Transform using Human First Impressions},
  author={Roygaga, Chaitanya and Krinsky, Joshua and Zhang, Kai and Kwok, Kenny and Bharati, Aparna},
  journal={arXiv preprint arXiv:2309.15381},
  year={2023}
}
```
