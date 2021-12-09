'''
Computer Vision Midterm Project
Project group members:  
    1. Aniket Bote (N12824308)
    2. Sindhu Harish (N19806874)
'''

import numpy as np

def grayscale_conversion(image):
    '''
        Converts colored image to grayscale
        Args:
            image: input is color image
        Return:
            image: grayscale converted image
    '''
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B     
    return  imgGray
