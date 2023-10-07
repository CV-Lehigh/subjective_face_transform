import argparse
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--orig_images_path", type=str, default='', help='Path to original set of images')
parser.add_argument("--mod_images_path", type=str, default='', help='Path to modified set of images')
parser.add_argument("--output_path", type=str, default='', help='Save path for combined results')
parser.add_argument('--attribute', type=str, default='', help='Selected attribute')
args = parser.parse_args()

os.makedirs(fr'{args.output_path}/{args.attribute}', exist_ok=True)
image_list = os.listdir(args.orig_images_path)

for img in tqdm(image_list):
    
    image_name = img.split('.')[0].split('_')[:-1]
    image_name = '_'.join([str(item) for item in image_name])
    orig_img = Image.open(fr'{args.orig_images_path}/{img}')
    
    edited_minus_2_img = Image.open(fr'{args.mod_images_path}/-0.2/{args.attribute}/images/{image_name}_inv_edit.jpg')
    edited_minus_1_img = Image.open(fr'{args.mod_images_path}/-0.1/{args.attribute}/images/{image_name}_inv_edit.jpg')
    edited_plus_1_img = Image.open(fr'{args.mod_images_path}/+0.1/{args.attribute}/images/{image_name}_inv_edit.jpg')
    edited_plus_2_img = Image.open(fr'{args.mod_images_path}/+0.2/{args.attribute}/images/{image_name}_inv_edit.jpg')
    
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, sharex=True, figsize=(52, 20), gridspec_kw = {'wspace':0, 'hspace':0})
    ax1.imshow(edited_minus_2_img)
    ax2.imshow(edited_minus_1_img)
    ax3.imshow(orig_img)
    ax4.imshow(edited_plus_1_img)
    ax5.imshow(edited_plus_2_img)
    
    plt.xticks([])
    plt.yticks([])
    
    fig.savefig(fr'{args.output_path}/{args.attribute}/{image_name}.jpg', bbox_inches="tight")
    plt.close('all')