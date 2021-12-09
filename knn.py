'''
Computer Vision Midterm Project
Project group members:  
    1. Aniket Bote (N12824308)
    2. Sindhu Harish (N19806874)
'''

import numpy as np
from collections import Counter

class KNNClassifier:
    '''
    Performs K-nearest neighbour classification using histogram intersection to compute similarity

    Args:
        k: The number of neighbours to consider
        X: The training features ie hog features 
        y: The training labels 
    '''
    def __init__(self, k, X, y):
        # Initialize all the parameters
        self.k = k
        self.X = X
        self.y = y
        self.n = X.shape[0]
        
    def compute_histogram_overlap(self, hist_values):
        '''
        Computes histogram overlap using formula:
        overlap = sum(min(I, M)) / sum(M)
        Args:
            hist_values: A single array consisting of values of histogram bins ie hog features
        Returns:
            histogram overlap between hist_values and all training features
        '''
        return np.sum(np.minimum(hist_values, self.X), axis = 1) / np.sum(self.X, axis = 1)
    
    def predict(self, X_test):
        '''
        Computes predictions
        Args:
            X_test: The testing features ie. hog features
        Return:
            preds: Predictions of testing features
            topk: list of tuple containing label, overlap value, index of top k features
        '''
        # Initialize empty array preds to store predictions, k-nearest neighbours
        preds = []
        topk = []


        # Iterate over test features
        for i in range(X_test.shape[0]):
            # Compute histogram overlap
            histogram_overlap = self.compute_histogram_overlap(X_test[i])
        
            # Create list of tuple containing label, overlap value, index ie [(1, 0.77, 0), (0, 0.55, 1), (1, 0.88, 2), (0, 0.99, 3)]
            similarity = list(zip(self.y, histogram_overlap, range(self.n)))

            # Sort the list of tuples according to overlap values ie [(0, 0.99, 3), (1, 0.88, 2), (1, 0.77, 0), (0, 0.55, 1)]
            similarity = sorted(similarity, key = lambda x: x[1], reverse = True)
            
            # Take the count the labels in top k tuples and sort it according to highest countie [(1, 2), (0,1)]
            k_predictions = sorted(Counter([label[0] for label in similarity[:self.k]]).items(), key = lambda x : x[1], reverse = True)

            # Add the label of highest count to prediction array
            preds.append(k_predictions[0][0])

            # Add top k neighbours to topk array
            topk.append(similarity[:self.k])
        
        # Return prediction array & topk array 
        return np.array(preds), topk
            

if __name__ == "__main__":
    X_train = np.array([
        [1,2,3,4,5,2,4,7,3],
        [7,2,1,3,7,3,9,1,4],
        [3,2,7,4,8,3,1,1,2],
        [9,9,9,9,9,9,9,9,9],
        [4,8,3,5,9,1,2,3,4]
    ])

    y_train = np.array([1,1,0,0,1])

    X_test = np.array([
        [3,7,8,4,6,1,1,9,0]
    ])

    knn_model = KNNClassifier(3, X_train, y_train)

    print(*zip(knn_model.compute_histogram_overlap(X_test[0]), y_train), sep="\n")
    
    print(knn_model.predict(X_test))      