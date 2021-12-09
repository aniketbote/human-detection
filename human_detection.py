'''
Computer Vision Midterm Project
Project group members:  
    1. Aniket Bote (N12824308)
    2. Sindhu Harish (N19806874)
'''

# Import the required libraries
from data import create_dataset
from knn import KNNClassifier
from output import generate_table

# Initialize path to train & test directories
TRAIN_DIR_POS = "data/Training images (Pos)"
TRAIN_DIR_NEG = "data/Training images (Neg)"
TEST_DIR_POS = "data/Test images (Pos)"
TEST_DIR_NEG = "data/Test images (Neg)"

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
output_df.to_csv("output.csv", index = False)


