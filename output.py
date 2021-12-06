import pandas as pd

def generate_table(y_true, y_pred, topk, train_image_list, test_image_list):
    '''
    Generate classification report
    Args:
        y_true: True labels
        y_pred: Predicted labels
        topk: Top k neighbours
        train_image_list: Names of images in training set
        test_image_list: Names of images in testing set
    Returns:
        df: The dataframe containing classification report
    '''
    # Labelmap to map integer values to string labels
    label_map = {0:"No-human", 1:"Human"}

    # Store the number of neighbours
    k = len(topk[0])

    # Initialize the dataframe
    df = pd.DataFrame()

    # Add names of test images in dataframe 
    df['Test image'] = test_image_list

    # Add labels of test images in dataframe
    df['Correct Classification'] = list(map(lambda x: label_map[x], y_true))

    # Iterate over lenght of k
    for i in range(k):
        # Initialize empty column
        col = []
        # Iterate over all rows in topk
        for row in topk:
            # For each row in topk list add the ith neighbours's name, overlap value & label to column
            col.append(f"{train_image_list[row[i][2]]}, {row[i][1]}, {label_map[row[i][0]]}")
        # Add the column in datatframe
        df[f'File name of {i+1} NN, distance & classification'] = col

    # Add the predicted labels in dataframe
    df[f'Classification from {k}-NN'] = list(map(lambda x: label_map[x], y_pred))

    # Return the dataframe
    return df

if __name__ == "__main__":
    y_pred = [0,1]
    y_true = [0,1]
    topk = [
        [(0, 0.8064516129032258, 2), (1, 0.7741935483870968, 0), (1, 0.717948717948718, 3)],
        [(1, 0.9064516129032258, 2), (1, 0.8741935483870968, 0), (1, 0.797948717948718, 3)]
    ]
    image_list = ["image0", "image1", "image2", "image3"]
    test_image_list = ["test0", "test1"]
    
    out_df = generate_table(y_true, y_pred, topk, image_list, test_image_list)
    out_df.to_csv("test.csv", index=False)
