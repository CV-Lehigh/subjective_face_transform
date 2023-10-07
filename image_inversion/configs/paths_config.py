dataset_paths = {
    #  Human Face (FFHQ - train , CelebA-HQ - test)
    "ffhq": "",
    "celeba_test": "",
    # Churches (LSUN Churches)
    "church_train": "",
    "church_test": "",
}

model_paths = {
    "w_encoder_ffhq": f"pretrained_dependencies/image_inversion_models/w_encoder_e4e_ffhq_encode.pt",
    "hyper_inverter_ffhq": f"pretrained_dependencies/image_inversion_models/hyper_inverter_e4e_ffhq_encode_large.pt",
    "stylegan2_ada_ffhq": f"pretrained_dependencies/image_inversion_models/stylegan2-ada-ffhq.pkl",
    "ir_se50": f"pretrained_dependencies/image_inversion_models/model_ir_se50.pth",
    "shape_predictor": f"pretrained_dependencies/image_inversion_models/shape_predictor_68_face_landmarks.dat",
    "resnet34": f"pretrained_dependencies/image_inversion_models/resnet34-333f7ec4.pth",
}