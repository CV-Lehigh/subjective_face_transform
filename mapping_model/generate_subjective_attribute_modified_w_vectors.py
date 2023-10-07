'''
original code (derived from): https://github.com/RameenAbdal/StyleFlow
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import time
import sys

import torch
import torch.optim as optim

import random
import module.flow as flow
import module.utils as utils
from module.utils import standard_normal_logprob

from module.dataset_play import WPlayDataset
import numpy as np

import pickle
from tqdm import tqdm

def modify_vectors(model, dataset, change_score, attribute, save_path, batch_size, device):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print('Finished loading...')
    
    model.eval()

    result = np.zeros((len(dataset), 512))

    with torch.no_grad():

        for i_batch, sample_batched in enumerate(tqdm(dataloader)):

            w = sample_batched['w'].float().to(device)
            
            org_avg = sample_batched['avg'].float().to(device)
            org_score = sample_batched['avg'].float().to(device)
            
            tar_score = sample_batched['tar_score'].float().to(device)
            
            old = org_score
            new = tar_score

            zero = torch.zeros(w.shape[0], 1).to(w)

            z, _ = model(w, old, zero)

            w_prime = model(z, new, zero, True)[0].cpu().numpy()

            result[i_batch*batch_size:(i_batch+1)*batch_size, :] = w_prime

    nf = 0
    for c in args.dims:
        if c == '-':
            nf += 1
    nf += 2
    os.makedirs(f'{save_path}/{change_score}', exist_ok=True)
    name = f'{change_score}/wdash_vectors_{attribute}.npy'
    np.save(os.path.join(save_path, name), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_model_path", type=str, default='', help='Trained flow mapping model')
    parser.add_argument("--mapping_model_train_dataset", type=str, default='', help='Dataset used to train flow mapping model')
    parser.add_argument("--w_latent_file_path", type=str, default='', help='Path to original w latent file')
    parser.add_argument("--prediction_file_path", type=str, default='', help='Path to original image predictions')
    parser.add_argument("--attribute", type=str, default='', help='Selected attribute')
    parser.add_argument('--dims', type=str, default='512-512-512') # 
    parser.add_argument("--num_blocks", type=int, default=4, help='Number of stacked CNFs.')
    parser.add_argument("--dim_ctx", type=int, default=1, help='Dimension of context: score prediction of selected subjective attribute')
    parser.add_argument("--batch_size", type=int, default=1, help='Batch size')
    parser.add_argument('--save_path', type=str, default='', help='Save path for modified w latent vectors')
    parser.add_argument('--gpu_id', type=int, default=0, help='Current GPU for processing')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    for change_score in ['+0.1', '+0.2', '-0.1', '-0.2']:
        model = flow.cnf(512, args.dims, args.dim_ctx, args.num_blocks)
        model = model.to(device)
        weight = torch.load(f'{args.mapping_model_path}/{args.attribute}_mapping_model_{args.mapping_model_train_dataset}.pt')
        model.load_state_dict(weight['state_dict'], strict = False)
        dataset = WPlayDataset(w_file=f'{args.w_latent_file_path}/w_save.npy', attribute_file=f'{args.prediction_file_path}/input_{args.attribute}.pkl', increment=float(change_score))
        modify_vectors(model, dataset, change_score, args.attribute, args.save_path, args.batch_size, device)