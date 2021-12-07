'''
Computer Vision Midterm Project
Project group members:  
    1. Aniket Bote (N12824308)
    2. Sindhu Harish (N19806874)
'''

import numpy as np
from utils import apply_discrete_convolution, Operator


def L2_normalize(vector, eps=1e-5):

    # normalise the vector using L2, incase if vector is a zero then eps will avoid the exception 
    return vector / np.sqrt(np.sum(vector ** 2) + eps ** 2)


def calculate_hog_feature_of_cell(number_of_orientations, gradient_magnitude, orientation_angle):
    """
    Compute 1 HOG feature of a cell. Return a row vector of size `n_orientations`
    """
    histogram_bin_width = int(180 / number_of_orientations)
    hog = np.zeros(number_of_orientations)
    for i in range(orientation_angle.shape[0]):
        for j in range(orientation_angle.shape[1]):
            orientation = orientation_angle[i, j]
            lower_bin_idx = int(orientation / histogram_bin_width)
            hog[lower_bin_idx] += gradient_magnitude[i, j]

    return hog / (gradient_magnitude.shape[0] * gradient_magnitude.shape[1])

# histogram_oriented_gradient_features



def histogram_oriented_gradient_features(image: np.ndarray,
                         n_orientations: int = 9, pixels_per_cell = (8, 8),
                         cells_per_block = (1, 1)) -> np.ndarray:
    """
    Compute HOG features of an image. Return a row vector
    """
    # gradient_Gx, gradient_Gy = compute_gradient(image)
    # g_row, g_col = _hog_channel_gradient(image)
    gradient_Gx = apply_discrete_convolution(image, Operator.gx)
    gradient_Gy = apply_discrete_convolution(image, Operator.gy)
    gradient_Gx = np.nan_to_num(gradient_Gx)
    gradient_Gy = np.nan_to_num(gradient_Gy)


    shape_along_x_axis, shape_along_y_axis = gradient_Gx.shape
    pixels_per_cell_x, pixels_per_cell_y = pixels_per_cell
    cells_per_block_x, cells_per_block_y = cells_per_block


    n_cells_along_x_axis = int(shape_along_x_axis / pixels_per_cell_x)  # number of cells along row-axis
    n_cells_along_y_axis = int(shape_along_y_axis / pixels_per_cell_y)  # number of cells along col-axis
    n_blocks_along_x_axis = (n_cells_along_x_axis - cells_per_block_x) + 1
    n_blocks_along_y_axis = (n_cells_along_y_axis - cells_per_block_y) + 1
    # now compute the histogram for each cell

    magnitudes = np.hypot(gradient_Gx, gradient_Gy)
    orientations = np.rad2deg(np.arctan2(gradient_Gx, gradient_Gy)) % 180

    # n_cells_along_x_axis = int(sx / pixels_per_cell_x) # Number of cells in x axis
    # n_cells_along_y_axis = int(sy / pixels_per_cell_y) # Number of cells in y axis
    # n_blocks_along_x_axis = int(n_cells_along_x_axis - cells_per_block_x) + 1
    # n_blocks_along_y_axis = int(n_cells_along_y_axis - cells_per_block_y) + 1

    hog_cells = np.zeros((n_cells_along_x_axis, n_cells_along_y_axis, n_orientations))

    x_value = 0
    # Compute HOG of each cell
    for it_x in range(n_cells_along_y_axis):
        y_value = 0
        for it_y in range(n_cells_along_x_axis):
            magnitudes_patch = magnitudes[y_value:y_value + pixels_per_cell_y, x_value:x_value + pixels_per_cell_x]
            orientations_patch = orientations[y_value:y_value + pixels_per_cell_y, x_value:x_value + pixels_per_cell_x]

            hog_cells[it_y, it_x] = calculate_hog_feature_of_cell(n_orientations, magnitudes_patch, orientations_patch)

            y_value += pixels_per_cell_y
        x_value += pixels_per_cell_x

    # hog_blocks_normalized = np.zeros((n_blocks_along_x_axis, n_blocks_along_y_axis, n_orientations))
    hog_blocks_normalized = np.zeros((n_blocks_along_x_axis, n_blocks_along_y_axis, cells_per_block_x, cells_per_block_y, n_orientations))
    # Normalize HOG by block
    for it_blocksx in range(n_blocks_along_y_axis):
        for it_blocky in range(n_blocks_along_x_axis):
            hog_block = hog_cells[it_blocky:it_blocky + cells_per_block_y, it_blocksx:it_blocksx + cells_per_block_x]
            hog_blocks_normalized[it_blocky, it_blocksx] = L2_normalize(hog_block)

    return hog_blocks_normalized.ravel()

