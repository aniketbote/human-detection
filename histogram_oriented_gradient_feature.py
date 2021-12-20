'''
Computer Vision Midterm Project
Project group members:  
    1. Aniket Bote (N12824308)
    2. Sindhu Harish (N19806874)
'''
# Import the required libraries
import numpy as np
from utils import apply_discrete_convolution, Operator

def L2_normalize(vector, eps=1e-5):
    
    '''
    Performs L2 normalisation of vector

    Args:
        vector: hog vector
        eps: eps to avoid exception
    '''
    return (vector / np.sqrt(np.sum(vector ** 2) + eps ** 2))

def calculate_hog_feature_of_cell(number_of_orientations, gradient_magnitude, orientation_angle):
    '''
    Compute gradient magnitude's corresponding histogram bin based on the orientation angle

    Args:
        number_of_orientations: number_of_orientations
        gradient_magnitude: vertical and horizontal gradient
        orientation_angle: angle corresponding to magnitude
    '''
    #Create and Initialise the hog vector of size 9(number of bins)
    hog = np.zeros(number_of_orientations)

    #Dictionary with index as key and bin centre as value
    histogram_bin = {0:10, 1:30, 2:50, 3:70, 4:90, 5:110, 6:130, 7:150, 8:170}

    #List with bin centre values, 190 is added for index access 
    bin_centre = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190]
    
    #Iterate over the gradient angle matrix
    for i in range(orientation_angle.shape[0]):
        for j in range(orientation_angle.shape[1]):

            #Access gradient angle
            orientation = orientation_angle[i, j]

            #Iterate over bin center array and assigning proper values to left and right bins
            for k in range(len(bin_centre)):
                if orientation <= bin_centre[k]:
                    left_bin = k - 1
                    if left_bin == -1:
                        left_bin = 0
                        right_bin = 8
                        break
                    if left_bin == 8:
                        right_bin = 0
                        break
                    right_bin = k
                    break
            
            #Gradient magnitude is between 170 and 180 degrees
            if orientation > 170 and orientation < 180:

                #Formula tpo split gradient magnitude between left and right bins
                right_val = ((180 - orientation) + 10) / 20* gradient_magnitude[i, j]
                left_val = gradient_magnitude[i, j] - right_val

                #Populate the hog vector with magnitude values
                hog[left_bin] += right_val
                hog[right_bin] += left_val
            
            #Gradient magnitude is between 0 and 10 degrees
            elif orientation > 0 and orientation < 10:

                 #Formula tpo split gradient magnitude between left and right bins
                left_val = (orientation + 10) / 20 * gradient_magnitude[i, j]
                right_val = gradient_magnitude[i, j] - left_val

                #Populate the hog vector with magnitude values
                hog[left_bin] += left_val
                hog[right_bin] += right_val
            
            #Gradient magnitude is greater than 10 and less than 170 degrees (general case)
            elif orientation > 10 and orientation < 170:

                #Formula tpo split gradient magnitude between left and right bins
                left_val= gradient_magnitude[i, j] * abs(histogram_bin[right_bin] - orientation) / 20
                right_val = gradient_magnitude[i, j] * abs(orientation - histogram_bin[left_bin]) / 20

                #Populate the hog vector with magnitude values
                hog[left_bin] += left_val
                hog[right_bin] += right_val
    return hog


def histogram_oriented_gradient_features(image: np.ndarray,
                         n_orientations: int = 9, pixels_per_cell = (8, 8),
                         cells_per_block = (1, 1)) -> np.ndarray:
    '''
    Calculate the Hog vector
    Args:
        image: input image 
        n_orientations: number of bins
        pixels_per_cell: number of pixels per cell
        cells_per_block: number of cells per block
    '''
    #Horizontal Gradient
    gradient_Gx = apply_discrete_convolution(image, Operator.gx)

    #Vertical Gradient
    gradient_Gy = apply_discrete_convolution(image, Operator.gy)
    
    #Remove Nan values with zeros
    gradient_Gx = np.nan_to_num(gradient_Gx)
    gradient_Gy = np.nan_to_num(gradient_Gy)

    #Taking shape along X and Y axis
    shape_along_x_axis, shape_along_y_axis = gradient_Gx.shape

    #Pixels_per_cell 8 * 8
    pixels_per_cell_x, pixels_per_cell_y = pixels_per_cell

    #Blocks per cell, here 2 * 2
    cells_per_block_x, cells_per_block_y = cells_per_block

    #number of cells along row-axis
    n_cells_along_x_axis = int(shape_along_x_axis / pixels_per_cell_x)  

    #number of cells along col-axis
    n_cells_along_y_axis = int(shape_along_y_axis / pixels_per_cell_y) 

    #number of blocks along X-axis
    n_blocks_along_x_axis = (n_cells_along_x_axis - cells_per_block_x) + 1

    #number of blocks along Y-axis
    n_blocks_along_y_axis = (n_cells_along_y_axis - cells_per_block_y) + 1
    
    #Gradient magnitude by adding Gx + Gy 
    magnitudes = abs(np.hypot(gradient_Gx, gradient_Gy))
    
    #Normalise gradient magnitude
    normalised_gradient = np.sqrt((3 * 255) ** 2 + (3 * 255) ** 2)
    normalised_gradient = (magnitudes/normalised_gradient) * 255
    
    #Gradient angle
    orientations = abs(np.rad2deg(np.arctan2(gradient_Gy, gradient_Gx)) % 180)

    #Initialising hog cells as numpy array
    hog_cells = np.zeros((n_cells_along_x_axis, n_cells_along_y_axis, n_orientations))

    x_value = 0

    # Compute HOG of each cell by iterating along X and Y cells
    for it_x in range(n_cells_along_y_axis):
        y_value = 0
        for it_y in range(n_cells_along_x_axis):
            #Calculate magnitude of cell or 8 * 8 pixels by adding vertical and horizontal gradient
            magnitudes_patch = normalised_gradient[y_value:y_value + pixels_per_cell_y, x_value:x_value + pixels_per_cell_x]

            #Calculate gradient angle of cells
            orientations_patch = orientations[y_value:y_value + pixels_per_cell_y, x_value:x_value + pixels_per_cell_x]
            
            #Calculate hog of cells
            hog_cells[it_y, it_x] = calculate_hog_feature_of_cell(n_orientations, magnitudes_patch, orientations_patch)

            y_value += pixels_per_cell_y
        x_value += pixels_per_cell_x

    # hog_blocks_normalized = np.zeros((n_blocks_along_x_axis, n_blocks_along_y_axis, n_orientations))
    hog_blocks_normalized = np.zeros((n_blocks_along_x_axis, n_blocks_along_y_axis, cells_per_block_x, cells_per_block_y, n_orientations))

    # Normalize HOG by block by iterating along X and Y axes
    for it_blocksx in range(n_blocks_along_y_axis):
        for it_blocky in range(n_blocks_along_x_axis):

            #calcualte hog block by using block size 16*16 pixels or (2*2 cells)
            hog_block = hog_cells[it_blocky:it_blocky + cells_per_block_y, it_blocksx:it_blocksx + cells_per_block_x]

            #Apply L2 normalization
            hog_blocks_normalized[it_blocky, it_blocksx] = L2_normalize(hog_block)

    return hog_blocks_normalized.ravel()


# if __name__ == "__main__":
#      sample = np.array([
#         [120, 125, 212, 239, 120, 125, 212, 239],
#         [90, 100, 180, 200, 120, 125, 212, 239],
#         [85, 195, 200, 210, 120, 125, 212, 239],
#         [75, 212, 255, 195, 120, 125, 212, 239],
#         [120, 125, 212, 239, 120, 125, 212, 239],
#         [90, 100, 180, 200, 120, 125, 212, 239],
#         [85, 195, 200, 210, 120, 125, 212, 239],
#         [75, 212, 255, 195, 120, 125, 212, 239]
#     ])

# output =  histogram_oriented_gradient_features(sample, n_orientations=9, pixels_per_cell=(8, 8),  cells_per_block=(2, 2))

