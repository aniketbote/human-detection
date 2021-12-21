'''
Computer Vision Final Project
Project group members:  
    1. Aniket Bote (N12824308)
    2. Sindhu Harish (N19806874)
'''
# Import the required libraries
import numpy as np
from utils import Operator, apply_discrete_convolution

def perform_gradient_operation(image):
    '''
    Args:
        image : An image on which gradient operation will happen
    Returns:
        Magnitude : Magnitude of the gradient
        Theta     : Gradient Angle
    '''
    # Compute horizontal gradients
    dfdx = apply_discrete_convolution(image, Operator.gx)

    # Compute vertical gradients
    dfdy = apply_discrete_convolution(image, Operator.gy)

    # Compute magnitude of the gradient
    m = np.sqrt(np.square(dfdx) + np.square(dfdy))
    
    # Normalize gradient magnitude
    m = np.nan_to_num((np.absolute(m) / (3 * 255.0)) * 255.0)

    # Compute gradient angle
    theta = np.nan_to_num(abs(np.degrees(np.arctan2(dfdy, dfdx))))

    return np.round(m), np.round(theta)

if __name__ == "__main__":
    from skimage.io import imread
    from perform_grayscale_conversion import grayscale_conversion
    # image =grayscale_conversion(imread("data\Training images (Pos)\crop_000010b.bmp"))
    np.random.seed(10)
    image = np.random.randint(0,255, (16,8))
    print(image) # gy = 282 gx = 88 norm = 295.4115773
    mag , ang = perform_gradient_operation(image)
    print(mag)
    print("&&&&")
    print(ang)

