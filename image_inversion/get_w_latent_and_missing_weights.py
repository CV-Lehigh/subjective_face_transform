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
from configs import data_configs  # noqa: E402
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
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input_path", type=str, default='', help='Path to aligned input images')
    parser.add_argument("--checkpoint_path", type=str, default='', help='Path to HyperInverter checkpoint (https://drive.google.com/file/d/1JxKAHk-u4joVq1NmDsVcR_ov-cNWFBSu/view)') # https://github.com/VinAIResearch/HyperInverter/blob/master/MODEL_ZOO.md
    args = parser.parse_args()
    
    out_path_results = os.path.join(args.input_path, "inv_images")
    out_path_results_weights = os.path.join(args.input_path, "added_weights")
    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_results_weights, exist_ok=True)

    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    opts = ckpt["opts"]
    opts.update(vars(args))
    opts = argparse.Namespace(**opts)
    
    # Load HyperInverter
    net = HyperInverter(opts)
    net.eval()
    net.cuda()
    
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args["transforms"](opts).get_transforms()
    dataset = InferenceDataset(root=f"{args.input_path}/images", transform=transforms_dict["transform_inference"], opts=opts)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(4), drop_last=False) # Keep batch_size=1

    num_images = len(dataset) # Update this variable to limit number of processed images

    w_save = np.empty([num_images, 512]) # Store all ws

    global_i = 0
    global_time = 0
    for input_batch in tqdm(dataloader): # Load images
        if global_i >= num_images:
            break
        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            w_images, final_images, predicted_weights, w_codes = run_on_batch(input_cuda, net)
            toc = time.time()
            global_time += toc - tic

        bs = final_images.size(0)
        for i in range(bs):
            final_image = tensor2im(final_images[i])
            im_path = dataset.paths[global_i]
            
            # Save w latent
            w_save[global_i, :] = w_codes[i][0].cpu().numpy()
            
            # Save added weights
            img_name_save = im_path.split('/')[-1].split('.')[0]
            im_added_weights_save_path = os.path.join(out_path_results_weights, os.path.basename(img_name_save))
            pred_weights_per_sample = {}
            for key in predicted_weights:
                pred_weights_per_sample[key] = predicted_weights[key][i]
            added_weights = common.convert_predicted_weights_to_dict(pred_weights_per_sample) # Convert to dict
            torch.save(added_weights, f'{im_added_weights_save_path}.pkl')
            
            Image.fromarray(np.array(final_image)).save(f'{out_path_results}/{img_name_save}_inv.jpg') # Save reconstructed image

            global_i += 1
          
    np.save(f'{args.input_path}/w_save.npy', w_save) # Save w latent for all input images
    
def run_on_batch(inputs, net):
    result_batch = net(inputs, return_latents=True)
    return result_batch

if __name__ == "__main__":
    run()
