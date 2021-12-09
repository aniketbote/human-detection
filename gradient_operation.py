'''
Computer Vision Midterm Project
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
    m = np.absolute(m) / 255.0

    # Compute gradient angle
    theta = abs(np.degrees(np.arctan2(dfdy, dfdx)))

    return m, theta
