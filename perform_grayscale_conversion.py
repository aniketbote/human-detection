'''
Computer Vision Midterm Project
Project group members:  
    1. Aniket Bote (N12824308)
    2. Sindhu Harish (N19806874)
'''

import numpy as np

def grayscale_conversion(image):
    image_height, image_width, channel = image.shape
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    # grayscale_converted_image = np.zeros((image_height, image_width))
    
    # for i in range(image_height):
    #     for j in range(image_width):
    #         grayscale_converted_image[i][j]= int(image[i][j][0] * 0.299 + image[i][j][1] * 0.587 + image[i][j][2]* 0.114)
    print(imgGray)      
    return  imgGray
