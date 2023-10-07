'''
note: code is inspired from
    https://github.com/mel-2445/Predicting-First-Impressions
'''

import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from os import path
import cv2
import threading
import pickle
import csv

import model as mod

def init(ckpt=None):

    if ckpt == None:
        raise Exception('No weights path provided!')
    
    # Initialize model
    num_classes = 1
    model = mod.PretrainedModel()
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.cuda()
    model.eval()

    return model

def read_image(img_path):
    img = cv2.imread(img_path)

def cache(start, img_list, num_of_images):
    print("start cacheing")
    cache = []
    for i in tqdm(range(num_of_images)):
        t = threading.Thread(target=read_image, args=(img_list[start+i],))
        cache.append(t)
        t.start()

    print("waiting thread")
    for t in tqdm(cache):
        t.join()


def predict(list_path, model):

    if not path.exists(list_path):
        raise Exception(f'Invalid file path: {list_path}')

    # Load image list file
    #img_list = np.asarray(pd.read_csv(list_path, header=None)).flatten()
    img_list = np.asarray(pd.read_csv(list_path, delimiter=' ', header=None)).flatten()

    result = np.empty([img_list.shape[0], 1])
    result_pickle = []
    
    # For csv
    filenames = []
    scores = []
    with torch.no_grad():

        for index in tqdm(range(len(img_list))):
            img_path = img_list[index]

            if not path.exists(img_path):
                print(f"{img_path} doesn't exist")
                continue

            img = cv2.imread(img_path)
            try:
                img = cv2.resize(img, (224, 224))
            except:
                print(img_path)
                continue

            x = torch.from_numpy(img).float().cuda() / 255.
            x = x[None, :, :, :]
            x = x.permute(0, 3, 1, 2)

            yhat = model(x)

            yhat = yhat.cpu().numpy().flatten()
            
            result_pickle.append([path.split(img_path)[1], yhat])
            result[index] = yhat
            
            # For csv
            filenames.append(path.split(img_path)[1])
            scores.append(yhat.tolist())
            
    return result, result_pickle, filenames, scores

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--io_type", type=str, default='', help='Input type: input, input_inv, output')
    parser.add_argument("--attribute", type=str, default='', help='Selected attribute')
    parser.add_argument('--list_path', type=str, default='', help='Path to the list of image path')
    parser.add_argument('--ckpt', type=str, default='', help='Path to trained attribute prediction model')
    parser.add_argument('--dest', type=str, default='', help='Path to folder to save attribute predictions')
    args = parser.parse_args()
    
    if args.attribute == 'Trustworthiness':
        CURR_MODEL = 'Trustworthiness__resnet18_AFLW_OMI' #'Trustworthiness__resnet18_AFLW_OMI_CFD, Trustworthiness__resnet18_AFLW_OMI'
    elif args.attribute == 'Dominance':
        CURR_MODEL = 'Dominance__resnet18_AFLW_OMI' #'Dominance__resnet18_AFLW_OMI_CFD, Dominance__resnet18_AFLW_OMI'
    elif args.attribute == 'Attractiveness':
        CURR_MODEL = 'Attractiveness__resnet18_SCUT_OMI' #'Attractiveness__resnet18_SCUT_OMI_CFD, Attractiveness__resnet18_SCUT_OMI'

    if args.io_type == 'input':
        score_list = ['']
    else:
        score_list = ['+0.1', '+0.2', '-0.1', '-0.2']
            
    for change_score in score_list:
        if args.io_type == 'output':
            list_path = fr'{args.list_path}/{change_score}/{args.attribute}/{args.io_type}_images_list.txt'
            dest_path = fr'{args.dest}/{change_score}'
        else:
            list_path = fr'{args.list_path}/{args.io_type}_images_list.txt'
            dest_path = fr'{args.dest}'
           
        if not path.exists(dest_path):
            os.makedirs(dest_path)

        model = init(fr'{args.ckpt}/{CURR_MODEL}.pt')
        result, result_pickle, filenames, scores = predict(list_path, model)

        ### For pickle file only
        result_pickle = sorted(result_pickle, key=lambda x:x[0])
        
        file_name = args.io_type + f'_{args.attribute}'
        
        # Save as numpy array
        np.save(f'{dest_path}/{file_name}.npy', result)
        
        # Save as pickle
        with open(f'{dest_path}/{file_name}.pkl', 'wb') as f:
            pickle.dump(result_pickle, f)
            
        # Save as csv
        all_scores = [list(item) for item in list(zip(filenames, scores))]
        with open(f"{dest_path}/{file_name}.csv", "a") as fp:
            writer = csv.writer(fp)
            writer.writerows(all_scores)