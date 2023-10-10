# Subjective Face Transform

Paper link: https://arxiv.org/abs/2309.15381

Abstract: Humans tend to form quick subjective first impressions of non-physical attributes when seeing someone's face, such as perceived trustworthiness or attractiveness. To understand what variations in a face lead to different subjective impressions, this work uses generative models to find semantically meaningful edits to a face image that change perceived attributes. Unlike prior work that relied on statistical manipulation in feature space, our end-to-end framework considers trade-offs between preserving identity and changing perceptual attributes. It maps identity-preserving latent space directions to changes in attribute scores, enabling transformation of any input face along an attribute axis according to a target change. We train on real and synthetic faces, evaluate for in-domain and out-of-domain images using predictive models and human ratings, demonstrating the generalizability of our approach. Ultimately, such a framework can be used to understand and explain biases in subjective interpretation of faces that are not dependent on the identity.

![teaser](teaser.png)

## Download and set-up environment

git clone https://github.com/CV-Lehigh/subjective_face_transform.git

cd subjective_face_transform

conda env create -f environment.yml

## Dependencies

Create a folder *pretrained_dependencies/* to add all pretrained model dependencies.

Face landmarks model: [shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)

Pretrained HyperInverter checkpoint: [hyper_inverter_e4e_ffhq_encode_large.pt](https://drive.google.com/file/d/1JxKAHk-u4joVq1NmDsVcR_ov-cNWFBSu/view)

Pretrained StyleGAN2-ADA: [stylegan2-ada-ffhq.pkl](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) {Please download and rename to *stylegan2-ada-ffhq.pkl*}

w-space encoder: [w_encoder_e4e_ffhq_encode.pt](https://drive.google.com/file/d/1uVqnXDBujAv4a4TU99SFIwKeAG-H6pzp/view)

irse50 encoder checkpoint: [model_ir_se50.pth](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view)

#### Inversion Dependencies

Here is a list of all pretrained models required to *project and invert* face images using the inversion setup. Please download and add all relevant models to the path *pretrained_dependencies/image_inversion_models*.

#### First Impression Prediction Models

Here is a list of our trained first impression prediction models, with a ResNet-18 base architecture. Training datasets for each model are presented. Empirically Chosen {EC} model for each attribute is as per results shown in the paper. Please download and add all relevant models to the path *pretrained_dependencies/subjective_models*.

Attractiveness: [AFLW](), [AFLW &#8594; (fine-tune) OMI {EC}]()

Dominance: [AFLW](), [AFLW &#8594; (fine-tune) OMI {EC}]()

Trustworthiness: [SCUT-FBP5500](), [SCUT-FBP5500 &#8594; (fine-tune) OMI {EC}]()

#### Mapping Models

Here is a list of our trained image⟷score mapping models, trained using the proposed pipeline. Training datasets for each model are presented. Empirically Chosen {EC} model for each attribute is as per results shown in the paper. Please download and add all relevant models to the path *pretrained_dependencies/mapping_models*.

Attractiveness: [FFHQ](), [StyleGAN2-ADA generated](), [CelebAMask-HQ {EC}]()

Dominance: [FFHQ](), [StyleGAN2-ADA generated](), [CelebAMask-HQ {EC}]()

Trustworthiness: [FFHQ](), [StyleGAN2-ADA generated](), [CelebAMask-HQ {EC}]()
