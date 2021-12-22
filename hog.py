'''
Computer Vision Final Project
Project group members:  
    1. Aniket Bote (N12824308)
    2. Sindhu Harish (N19806874)
'''
import numpy as np
from gradient_operation import perform_gradient_operation

class HOG:
    def __init__(self, n_bins, cell_size, block_size, step_size, max_m = 180):
        '''
        Initialize the HOG class
        Args:
            n_bins: Number of bins
            cell_size: A tuple containing cell size in pixels
            block_size: A tuple containing block size using cells
            step_size: The step to take in order to create overlapping blocks (Using cells)
            max_m: Maximum allowed angle
        '''
        self.n_bins = n_bins
        self.bin_range = max_m / n_bins 
        self.cell_size = cell_size
        self.block_size = block_size
        self.step_size = step_size
    
    def __call__(self, img):
        '''
        Function that returns hog descriptor
        Args:
            img: The grayscale image
        Returns:
            HOG descriptor array
        '''
        # Computes gradient magnitude and gradient angle
        gradient_magnitude, gradient_angle = perform_gradient_operation(img)
        return self.compute_hog_features(gradient_magnitude, gradient_angle)

    def get_bins_and_fraction(self, g):
        '''
        Function to compute the bin numbers and fraction of magintude that goes into respectective bin
        Args:
            g: Gradient angle
        Returns:
            bin_i: Left bin number
            bin_j: Right bin number
            fraction_i = Fraction of magnitude that goes into left bin
            fraction_j = Fraction of magnitude that goes into right bin
        '''
        # Covert the gradient angle range from 0-180 --> 10 - 190 and divide the angle by 20 to efficintly calculate the right bin. j ranges from 0-9
        # This is done to compute the fraction with ease
        j = int(np.floor((g + self.bin_range/2) / self.bin_range))
        # Left bin --> Right bin - 1. i ranges from -1 - 8
        i = int(j - 1)
        # Fraction for left bin --> distance of g from right bin center/ bin range
        fraction_i = ((self.bin_range * j + self.bin_range/2) - g) / self.bin_range
        # Fraction for right bin --> distance of g from left bin center/ bin range
        fraction_j = (g - (self.bin_range * i + self.bin_range/2)) / self.bin_range
        # Use modulo operator to compute the true bin value
        bin_j = j  % self.n_bins
        # Use modulo operator to compute the true bin value
        bin_i = (bin_j - 1) % self.n_bins
        return bin_i, bin_j, fraction_i, fraction_j
    
    def l2_normalize(self, vector, epsilon=1e-5):
        '''
        Performs L2 normalization of vector

        Args:
            vector: hog block vector
            epsilon: epsilon to avoid exception

        Returns:
            Normalized vector
        '''
        return vector / np.sqrt(np.sum(vector ** 2) + epsilon ** 2)

    def compute_hog_cell(self, m_cell, g_cell):
        '''
        Function to compute hog features for a cell
        Args:
            m_cell: Cell containing maginitude 
            g_cell: Cell containing gradient angle
        Returns:
            Hog feature array for a cell
        '''
        # Initialize empty hog array
        hog_values = np.zeros((self.n_bins))

        # Iterate through range of all magnitude and gradient angles
        for i in range(m_cell.shape[0]):
            for j in range(m_cell.shape[1]):
                # Get the bins and fractions for gradient value
                bin_1, bin_2, fraction_1, fraction_2 = self.get_bins_and_fraction(g_cell[i][j])
                # Add the magnitude to respective bins based of fractions
                hog_values[bin_1] = hog_values[bin_1] + m_cell[i][j] * fraction_1
                # Add the magnitude to respective bins based of fractions
                hog_values[bin_2] = hog_values[bin_2] + m_cell[i][j] * fraction_2
        return hog_values

    def compute_hog_features(self, gradient_magnitude, gradient_angle):
        '''
        Function to compute hog features
        Args:
            gradient_magnitude: The gradient magnitude
            gradient_angle: The gradient angles
        '''
        # compute height and width of image
        height, width = gradient_magnitude.shape
        # compute number of cells in the image
        n_cells_x, n_cells_y =  int(height / self.cell_size[0]), int(width / self.cell_size[1])
        # compute numbr of blocks in the image 
        n_blocks_x, n_blocks_y = int(((n_cells_x - self.block_size[0]) / self.step_size) + 1), int(((n_cells_y - self.block_size[1]) / self.step_size) + 1)

        # Initialize zeros for all cell
        hog_cells = np.zeros((n_cells_x, n_cells_y, self.n_bins))
        # Iterate over the range of all cells in image
        for x in range(n_cells_x):
            for y in range(n_cells_y):
                # Get the cell for gradient magnitude
                m_block = gradient_magnitude[x * self.cell_size[0]: (x + 1) * self.cell_size[0], y * self.cell_size[1]: (y + 1) * self.cell_size[1]]
                # Get the cell for gradient angle
                g_block = gradient_angle[x * self.cell_size[0]: (x + 1) * self.cell_size[0], y * self.cell_size[1]: (y + 1) * self.cell_size[1]]
                # Compute hog features for cell
                hog_values_cell = self.compute_hog_cell(m_block, g_block)
                # Assign the hog features to respective cell
                hog_cells[x, y] = hog_values_cell
        
        # Initialize empty hog descriptor
        hog_descriptor = []
        # Iterate over the range of the blocks
        for x in range(n_blocks_x):
            for y in range(n_blocks_y):
                # Get the hog features for all the cells included in 1 block
                block = hog_cells[x : x + self.block_size[0], y : y + self.block_size[1]]
                # Flatten the hog features into vector, apply L2 normalization and append into the hog descriptor
                hog_descriptor += list(self.l2_normalize(block.ravel()))
        # Return the hog descriptor
        return np.array(hog_descriptor)

    

if __name__ == "__main__":
    np.random.seed(10)
    print("Test for get bins and fractions")
    h = HOG(9, (8,8), (2,2), 1, 180)
    GA = list(range(0,181,5))
    for a in GA:
        i, j, fi, fj = h.get_bins_and_fraction(a)
        print( f"{a} : {i, j} Bin centers: {h.bin_range * i  + h.bin_range / 2} & {h.bin_range * j  + h.bin_range / 2} Fraction : {fi, fj}")
    
    # HW example 4a: 
    # Answer = 29160
    print("\nTest for shape")
    h = HOG(18, (8,8), (3,3), 2, 360)
    image = np.random.randint(0,255, (296, 168))
    print(h(image).shape)

    # HW example 4b: Note the bin centers are different in HW
    # Answer [ 45.  55. 165.   0.   0.   0.   0.   0.   0.  30.  90.   0.   0.   0.   0.   0.   0. 135.]
    print("\nTest for hog cell values")
    h = HOG(18, (8,8), (1,1), 1, 360)
    GA = np.array([
        [200, 45, 23, 98, 130, 260, 255, 250],
        [125, 295, 85, 90, 130, 265, 249, 240],
        [123, 35, 85, 95, 125, 260, 250, 240],
        [100, 90, 45, 90, 120, 265, 240, 230],
        [95, 99, 105, 106, 355, 120, 100 ,110],
        [90, 205, 110, 120, 120, 130, 125, 120],
        [85, 90, 100, 110, 110, 120, 120, 110],
        [80, 80, 100, 110, 100, 100, 100, 110]])
    
    M = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 220, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 180, 0, 0, 0], 
        [0, 120, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0]])
    
    print(h.compute_hog_cell(M, GA))

    # Answer = [0.63496502 0.08945307 0.02757537 0.         0.         0.03941083 0.02796246 0.10478557 0.75811744]
    print("\nTest with sample image array")
    h = HOG(9, (8,8), (1,1), 1, 180)
    image = np.array([
        [120, 125, 212, 239, 120, 125, 212, 239],
        [90, 100, 180, 200, 120, 125, 212, 239],
        [85, 195, 200, 210, 120, 125, 212, 239],
        [75, 212, 255, 195, 120, 125, 212, 239],
        [120, 125, 212, 239, 120, 125, 212, 239],
        [90, 100, 180, 200, 120, 125, 212, 239],
        [85, 195, 200, 210, 120, 125, 212, 239],
        [75, 212, 255, 195, 120, 125, 212, 239]
    ])
    print(h(image))

    # Answer = 7524
    print("\nTest with sample image array with original dimension")
    h = HOG(9, (8,8), (2,2), 1, 180)
    image = np.random.randint(0,255, (160, 96))
    print(h(image).shape)

    #Answer = 19584
    print("\nTest with sample image array with uneven cell size and block size")
    h = HOG(9, (4,6), (8,2), 2, 180)
    image = np.random.randint(0,255, (160, 96))
    print(h(image).shape)


    