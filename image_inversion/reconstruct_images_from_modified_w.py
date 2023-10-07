'''
original code (derived from): https://github.com/VinAIResearch/HyperInverter
'''

import argparse
import os
import sys
import time
import pickle

sys.path.append(".")
sys.path.append("..")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as T
from configs import paths_config  # noqa: E402
from datasets.inference_dataset import InferenceDataset  # noqa: E402
from models.hyper_inverter import HyperInverter  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402
from utils import common
from utils.common import log_input_image, tensor2im, convert_predicted_weights_to_dict  # noqa: E402
from utils.log_utils import get_concat_h  # noqa: E402

from evaluation.latent_creators import (  # noqa: E402
    E4ELatentCreator,
    HyperInverterLatentCreator,
    PSPLatentCreator,
    ReStyle_E4ELatentCreator,
    SG2LatentCreator,
    SG2PlusLatentCreator,
    WEncoderLatentCreator,
)
from models.stylegan2_ada import Generator  # noqa: E402

def run():
    parser = argparse.ArgumentParser('')
    parser.add_argument("--checkpoint_path", type=str, default='', help='Path to HyperInverter checkpoint (https://drive.google.com/file/d/1JxKAHk-u4joVq1NmDsVcR_ov-cNWFBSu/view)') # https://github.com/VinAIResearch/HyperInverter/blob/master/MODEL_ZOO.md
    parser.add_argument("--input_image_list_path", type=str, default='', help='Path to list of input images')
    parser.add_argument("--w_latent_file_path", type=str, default='', help='Path to modified w latent file')
    parser.add_argument("--added_weights_file_path", type=str, default='', help='Additional information from input images for identity preservation of original faces')
    parser.add_argument("--stylegan2_ada_path", type=str, default='', help='Path to generator model')
    parser.add_argument("--attribute", type=str, default='', help='Selected attribute')
    parser.add_argument('--save_path', type=str, default='', help='Save path for modified (reconstructed) images')
    args = parser.parse_args()
    
    for change_score in ['+0.1', '+0.2', '-0.1', '-0.2']:
        
        # Load input image paths
        input_image_list_file = open(f'{args.input_image_list_path}/input_images_list.txt', "r")
        data = input_image_list_file.read()
        input_image_list = data.replace('\n', ' ').split(".")
        input_image_list_file.close()
        
        # Define save path
        inverted_image_output_path = f"{args.save_path}/{change_score}/{args.attribute}/images"
        os.makedirs(inverted_image_output_path, exist_ok=True)
        
        # Load modified w vectors file
        w_save = np.load(f'{args.w_latent_file_path}/{change_score}/wdash_vectors_{args.attribute}.npy')
        
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        opts = ckpt["opts"]
        opts.update(vars(args))
        opts = argparse.Namespace(**opts)
        
        # Load StyleGAN2-ada
        model_path = args.stylegan2_ada_path
        with open(model_path, "rb") as f:
            G_ckpt = pickle.load(f)["G_ema"]
            G_ckpt = G_ckpt.float()
        G = Generator(**G_ckpt.init_kwargs)
        G.load_state_dict(G_ckpt.state_dict())
        G.cuda().eval()
        
        for w, orig_image_path in zip(w_save, input_image_list):
            img_name = orig_image_path.split('/')[-1]
            
            w = torch.from_numpy(np.expand_dims(np.expand_dims(w, axis=0).repeat(18, axis=0), axis=0)).float().cuda()
            weights = torch.load(f'{args.added_weights_file_path}/{img_name}.pkl', map_location=torch.device('cuda')) # Load added weight for image
            inverted_image = G.synthesis(w, added_weights=weights, noise_mode="const")[0]
            inverted_image = tensor2im(inverted_image)
            
            inverted_image.save(f'{inverted_image_output_path}/{img_name}_inv_edit.jpg') # Save reconstructed image

if __name__ == "__main__":
    run()
