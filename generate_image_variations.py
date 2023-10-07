'''
Info: Run file to generate variations of input set of images along the score range of [-0.2,0.2], with generation of corresponding attribute predictions.
Note: Please refer execution files for details on input arguments
'''

import subprocess
import os
import sys

''' User-defined '''
CURRENT_DIR = os.getcwd()
ORIG_IMAGE_PATH = fr"sample_images_FFHQ/" # Provide path to raw input images (folder can have 1 or more images)
ATTRIBUTE = "Attractiveness" # options: Attractiveness, Dominance, Trustworthiness

###############################################################################################################################################

print("\n")
print("*********  Align input images  *********")
subprocess.run(["python", fr"image_inversion/align_all_parallel.py", 
                "--raw_dir", fr"{ORIG_IMAGE_PATH}", 
                "--saved_dir", fr"input_aligned/images/",
                "--shape_predictor_path", fr"pretrained_dependencies/image_inversion_models/shape_predictor_68_face_landmarks.dat"])

print("\n")
print("*********  Get image latents  *********")
subprocess.run(["python", fr"image_inversion/get_w_latent_and_missing_weights.py",
                "--input_path", fr"input_aligned",
                "--checkpoint_path", fr"pretrained_dependencies/image_inversion_models/hyper_inverter_e4e_ffhq_encode_large.pt"])

print("\n")
print("*********  Generate input image list for further processing  *********")
subprocess.run(["python", fr"first_impression_prediction/generate_image_list.py",
                "--main_folder_path", fr"input_aligned",
                "--io_type", "input", # options: input, input_inv, output
                "--attribute", f"{ATTRIBUTE}"]) 

print("\n")
print("*********  Get first impression scores corresponding to images  *********")
subprocess.run(["python", "first_impression_prediction/predict_subjective_attribute.py",
                "--io_type", "input", # options: input, input_inv, output
                "--attribute", f"{ATTRIBUTE}",
                "--list_path", fr"input_aligned",
                "--ckpt", fr"pretrained_dependencies/subjective_models",
                "--dest", fr"input_aligned/impression_predictions"])

print("\n")
print("*********  Get transformed w vectors (latents) over a score spectrum of [-0.2,0.2]  *********")
subprocess.run(["python", "mapping_model/generate_subjective_attribute_modified_w_vectors.py",
                "--mapping_model_path", fr"pretrained_dependencies/mapping_models",
                "--mapping_model_train_dataset", "celebamaskhq", # options: ffhq (Real), sg2ada_gen (Syn), celebamaskhq (Real)
                "--w_latent_file_path", fr"input_aligned",
                "--prediction_file_path", fr"input_aligned/impression_predictions",
                "--attribute", f"{ATTRIBUTE}",
                "--save_path", fr"output/modified_w_latents",
                "--gpu_id", "3"])

print("\n")
print("*********  Reconstruct images from transformed w vectors (latents)  *********")
subprocess.run(["python", "image_inversion/reconstruct_images_from_modified_w.py",
                "--checkpoint_path", fr"pretrained_dependencies/image_inversion_models/hyper_inverter_e4e_ffhq_encode_large.pt",
                "--input_image_list_path", fr"input_aligned",
                "--w_latent_file_path", fr"output/modified_w_latents",
                "--added_weights_file_path", fr"input_aligned/added_weights",
                "--stylegan2_ada_path", fr"pretrained_dependencies/image_inversion_models/stylegan2-ada-ffhq.pkl",
                "--attribute", f"{ATTRIBUTE}",
                "--save_path", fr"output/modified_images"])

print("\n")
print("*********  Generate output image list of modified images for further processing  *********")
subprocess.run(["python", "first_impression_prediction/generate_image_list.py",
                "--main_folder_path", fr"output/modified_images",
                "--io_type", "output", # options: input, input_inv, output
                "--attribute", f"{ATTRIBUTE}"])

print("\n")
print("*********  Get first impression scores corresponding to modified images  *********")
subprocess.run(["python", "first_impression_prediction/predict_subjective_attribute.py",
                "--io_type", "output", # options: input, input_inv, output
                "--attribute", f"{ATTRIBUTE}",
                "--list_path", fr"output/modified_images",
                "--ckpt", fr"pretrained_dependencies/subjective_models",
                "--dest", fr"output/impression_predictions"])

print("\n")            
print("*********  Combining modified images  *********")
subprocess.run(["python", "image_inversion/get_combined_image_of_edited_variants.py",
                "--orig_images_path", "input_aligned/inv_images",
                "--mod_images_path", "output/modified_images",
                "--output_path", f"output/modified_images/combined_variants",
                "--attribute", f"{ATTRIBUTE}"])