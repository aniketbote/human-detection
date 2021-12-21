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
    maximum_gradient_magnitude = np.sqrt((3 * 255.0)**2 + (3 * 255.0)**2)
    print(maximum_gradient_magnitude)
    m = np.nan_to_num((np.absolute(m) / maximum_gradient_magnitude) * 255.0)

    # Compute gradient angle, convert the range from [-180, 180] --> [0, 360] --> [0,180]
    theta = ((np.nan_to_num(np.degrees(np.arctan2(dfdy, dfdx))) + 360) % 360) % 180
    
    return np.round(m), theta

if __name__ == "__main__":
    from skimage.io import imread
    from grayscale import convert_to_grayscale
    # image =convert_to_grayscale(imread("data\Training images (Pos)\crop_000010b.bmp"))
    np.random.seed(10)
    image = np.random.randint(0,255, (16,8))
    print(image) # 1,1 -- gx = 282 gy = 88 norm = 69.62 --> 70  1,2 -- gx = -162 gy = -152 norm = 52.35 --> 52 angle = 43.17
    print("****")
    mag , ang = perform_gradient_operation(image)
    print(mag)
    print("&&&&")
    print(ang)

