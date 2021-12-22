'''
Computer Vision Final Project
Project group members:  
    1. Aniket Bote (N12824308)
    2. Sindhu Harish (N19806874)
'''

import numpy as np

# A class to store all operators
class Operator:
    # Prewitt operator for Gx
    gx = np.array([
        [-1,0,1],
        [-1,0,1],
        [-1,0,1]])
    
    # Prewitt operator for Gy
    gy = np.array([
        [1,1,1],
        [0,0,0],
        [-1,-1,-1]])

# A function to apply dicreet convolutions
def apply_discrete_convolution(image, mask):
    '''
    Args:
        image : An image to use for convolution
        mask  : An mask to use for convolution
    Returns:
        convolved image: An image after convolution
    '''
    # Get the shape of image and mask
    (m_image, n_image), (m_mask, n_mask) = image.shape, mask.shape

    # Compute the reference pixel index from where output array will start populating
    rpi_m, rpi_n = int(np.floor(m_mask/2)), int(np.floor(n_mask/2))

    # Initialize an output array with nan values
    output_arr = np.ones((m_image, n_image)) * np.nan

    # Iterate through the image
    for i in range(m_image - m_mask + 1):
        for j in range(n_image - n_mask + 1):
            # Isolate the image slice to apply convolution
            img_slice = image[i:i+m_mask, j:j+n_mask]
            # Apply convolution and store the result in output array in approriate location
            output_arr[i+rpi_m][j+rpi_n] = np.sum(img_slice * mask)

    return output_arr
