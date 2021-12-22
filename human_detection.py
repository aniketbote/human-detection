'''
Computer Vision Final Project
Project group members:  
    1. Aniket Bote (N12824308)
    2. Sindhu Harish (N19806874)
'''

# Import the required libraries
import os
import shutil
import numpy as np

from data import create_dataset
from knn import KNNClassifier
from output import generate_table, save_normalized_images


# Initialize variables
# Path to train test directories
TRAIN_DIR_POS = "data/Training images (Pos)" 
TRAIN_DIR_NEG = "data/Training images (Neg)"
TEST_DIR_POS = "data/Test images (Pos)"
TEST_DIR_NEG = "data/Test images (Neg)"

# Path to save the hog values
HOG_SAVE_PATH = "Output/hog_values"

# Path to save the normalized images
IMAGE_SAVE_PATH = "Output/normalized_images"

# Path to save the output table
TABLE_SAVE_PATH = "Output/output.csv"

# Names of images to compute hog values
FILE_NAME_TRAIN = ["crop001028a.bmp", "crop001030c.bmp", "00000091a_cut.bmp"]
FILE_NAME_TEST = ["crop001278a.bmp", "crop001500b.bmp", "00000090a_cut.bmp"]

# Remove the output folder if it exists
if os.path.exists(HOG_SAVE_PATH):
    shutil.rmtree(HOG_SAVE_PATH)

# Remove the output folder if it exists
if os.path.exists(IMAGE_SAVE_PATH):
    shutil.rmtree(IMAGE_SAVE_PATH)

# Create the output folder
os.makedirs(HOG_SAVE_PATH)
os.makedirs(IMAGE_SAVE_PATH)


# Initialize the number of neighbours
K = 3

# Create dataset from train & test images
X_train, y_train, train_image_list = create_dataset(TRAIN_DIR_POS,TRAIN_DIR_NEG)
X_test, y_test, test_image_list = create_dataset(TEST_DIR_POS, TEST_DIR_NEG)

# Create KNN classfier model
knn_classfier = KNNClassifier(K,X_train, y_train)

# Use the KNN classifier model for predictions
y_preds, k_top_neighbours = knn_classfier.predict(X_test)

# Generate output table & save the results
output_df = generate_table(y_test, y_preds, k_top_neighbours, train_image_list, test_image_list)
output_df.to_csv(TABLE_SAVE_PATH, index = False)

# Saving the normalized test images
save_normalized_images(TEST_DIR_POS, TEST_DIR_NEG, IMAGE_SAVE_PATH)

# Save the hog values for selected images
for image_name in FILE_NAME_TEST:
    index = test_image_list.index(image_name)
    np.savetxt(os.path.join(HOG_SAVE_PATH,"hog_"+ image_name.split('.')[0] + '.txt'), X_test[index])

for image_name in FILE_NAME_TRAIN:
    index = train_image_list.index(image_name)
    np.savetxt(os.path.join(HOG_SAVE_PATH,"hog_"+ image_name.split('.')[0] + '.txt'), X_train[index])






