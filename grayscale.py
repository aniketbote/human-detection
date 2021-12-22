'''
Computer Vision Final Project
Project group members:  
    1. Aniket Bote (N12824308)
    2. Sindhu Harish (N19806874)
'''

import numpy as np

def convert_to_grayscale(image):
    '''
        Converts colored image to grayscale
        Args:
            image: input is color image
        Return:
            image: grayscale converted image
    '''
    # Segregate R,G,B channels
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    # Apply grayscale conversion to R, G, B channel
    imgGray = np.round(0.299 * R + 0.587 * G + 0.114 * B)   
    return  imgGray

if __name__ == "__main__":
    from skimage.io import imread
    image =imread("data\Training images (Pos)\crop_000010b.bmp")
    print(convert_to_grayscale(image))


