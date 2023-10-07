import os
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--main_folder_path", type=str, default='', help='Path to image folder')
parser.add_argument("--io_type", type=str, default='', help='Input type: input, input_inv, output')
parser.add_argument("--attribute", type=str, default='', help='Selected attribute')
args = parser.parse_args()

if args.io_type == 'output':
    score_list = ['+0.1', '+0.2', '-0.1', '-0.2']
else:
    score_list = ['']
for change_score in score_list:
    if args.io_type == 'output':
        path = fr'{args.main_folder_path}/{change_score}/{args.attribute}'
    else:
        path = fr'{args.main_folder_path}'
    if args.io_type == 'input_inv':
        image_list = os.listdir(f'{path}/inv_images')
    else:
        image_list = os.listdir(f'{path}/images')
    image_list.sort()

    file = open(f'{path}/{args.io_type}_images_list.txt', "w")

    for img in image_list:
        if args.io_type == 'input_inv':
            name = f'{path}/inv_images/{img}\n'
        else:
            name = f'{path}/images/{img}\n'
        file.write(name)
    file.close()