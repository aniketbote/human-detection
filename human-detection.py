'''
Computer Vision Project
Project group members:  
    1. Aniket Bote (N12824308)
    2. Sindhu Harish (N19806874)
'''

# Import the required libraries
import argparse
import glob
import os
import numpy as np
from skimage.feature import hog

import cv2

from perform_grayscale_conversion import grayscale_conversion
from gradient_operation import perform_gradient_operation
from histogram_oriented_gradient_feature import histogram_oriented_gradient_features


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_folder',
    type=str,
    default='training_images',
    required=False,
    help='input folder with images'                        
)

# parser.add_argument(
#     '--output_folder',
#     type=str,
#     default='output',
#     required=False,
#     help='output folder to save processed images'                        
# )

args = parser.parse_args()

# if os.path.exists(args.output_folder):
#     shutil.rmtree(args.output_folder)

# os.makedirs(args.output_folder)

print("Reading images from input folder")
images = glob.glob(os.path.join(args.input_folder, '*.bmp'))
for image_name in images:
    output_image_name = image_name.split('\\')[1].split('.bmp')[0]
    img = cv2.imread(image_name)   

    print("Performing conversion to grayscale of the image: " + output_image_name)
    grayscale_conversion_image = grayscale_conversion(img)
   
    print("Performing gradient smoothing for image: " + output_image_name)
    M, THETA = perform_gradient_operation(grayscale_conversion_image)

    print("Performing hog for image: " + output_image_name)
    hog_features = histogram_oriented_gradient_features(grayscale_conversion_image, n_orientations=9,
    pixels_per_cell=(8, 8),  cells_per_block=(2, 2))

    hog_features_check = hog(
        img, orientations=9,
        pixels_per_cell=(8, 8), cells_per_block=(2, 2),
        block_norm='L2')

assert hog_features.shape == hog_features_check.shape
print(np.allclose(hog_features, hog_features_check))
print(hog_features.shape)

print('check')
   
    # print("Performing non-maxima suppression for image: " + output_image_name)
    # NMS = perform_non_maxima_suppression(args, output_image_name, M, THETA)

    # print("Performing thresholding for image: " + output_image_name, '\n')
    # T1, T2, T3 = perform_thresholding(args, output_image_name, NMS)


