'''
Computer Vision Final Project
Project group members:  
    1. Aniket Bote (N12824308)
    2. Sindhu Harish (N19806874)
'''

# Import the required libraries
import os
import numpy as np
from skimage.io import imread
from hog import HOG
from grayscale import convert_to_grayscale 

def load_data(path):
    '''
    Read image data from directory and compute hog features

    Args: 
        path: The path string to image dir
    Returns:
        matrix: An array containing the hog features of all images in image dir
        image_list: An list containing all the image names
    '''

    # Initialize empty array matrix, image_list
    matrix = []
    image_list = []

    # Initialize HOG object 
    hog_obj = HOG(n_bins=9, cell_size=(8,8), block_size=(2,2), step_size=1)

    # Iterate over all the images in path
    for image in os.listdir(path):
        # Read and convert the images to grayscale
        img = convert_to_grayscale(imread(os.path.join(path,image)))

        # Compute hog features
        fd = hog_obj(img)

        # Add hog features to matrix
        matrix.append(fd)
        
        # Add image name to image list
        image_list.append(image)


    # Return matrix, image list
    return np.array(matrix), image_list

def create_dataset(pos_path, neg_path):
    '''
    Args:
        pos_path: The image directory containing positive images
        neg_path: The image directory containing negative images
    Returns:
        X: Features of all the images
        y: Labels of all images

    '''
    # Compute hog features
    X_pos, x_pos_names = load_data(pos_path)

    # Assign label 1 for positive images
    y_pos = np.ones((X_pos.shape[0]))
    
    # Compute hog features
    X_neg, x_neg_names = load_data(neg_path)

    # Assign label 0 for negative images
    y_neg = np.zeros((X_neg.shape[0]))

    # Concatenate all the features/ lables / names
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    image_list = x_pos_names + x_neg_names

    # Return the labels
    return X, y, image_list


if __name__ == "__main__":
    TRAIN_DIR_POS = "data/Training images (Pos)"
    TRAIN_DIR_NEG = "data/Training images (Neg)"
    X_train, y_train, image_list = create_dataset(TRAIN_DIR_POS, TRAIN_DIR_NEG)
    print(X_train.shape, y_train.shape)
    print(*image_list, sep = '\n')
