import numpy as np
from gradient_operation import perform_gradient_operation

class HOG:
    def __init__(self, n_bins, cell_size, block_size, step_size, max_m = 180):
        self.n_bins = n_bins
        self.bin_range = max_m / n_bins 
        self.cell_size = cell_size
        self.block_size = block_size
        self.step_size = step_size
    
    def __call__(self, img):
        gradient_magnitude, gradient_angle = perform_gradient_operation(img)
        return self.compute_hog_features(gradient_magnitude, gradient_angle)

    def get_bins_and_fraction(self, g):
        j = int(np.floor((g + self.bin_range/2) / self.bin_range))
        i = int(j - 1)
        fraction_i = ((self.bin_range * j + self.bin_range/2) - g) / self.bin_range
        fraction_j = (g - (self.bin_range * i + self.bin_range/2)) / self.bin_range
        bin_j = j  % self.n_bins
        bin_i = (bin_j - 1) % self.n_bins
        return bin_i, bin_j, fraction_i, fraction_j
    
    def l2_normalize(self, vector, epsilon=1e-5):
        '''
        Performs L2 normalization of vector

        Args:
            vector: hog block vector
            epsilon: epsilon to avoid exception
        '''
        return vector / np.sqrt(np.sum(vector ** 2) + epsilon ** 2)

    def compute_hog_cell(self, m_cell, g_cell):
        hog_values = np.zeros((self.n_bins))
        for i in range(m_cell.shape[0]):
            for j in range(m_cell.shape[1]):
                bin_1, bin_2, fraction_1, fraction_2 = self.get_bins_and_fraction(g_cell[i][j])
                hog_values[bin_1] = hog_values[bin_1] + m_cell[i][j] * fraction_1
                hog_values[bin_2] = hog_values[bin_2] + m_cell[i][j] * fraction_2
        return hog_values

    def compute_hog_features(self, gradient_magnitude, gradient_angle):
        height, width = gradient_magnitude.shape
        n_cells_x, n_cells_y =  int(height / self.cell_size[0]), int(width / self.cell_size[1])
        n_blocks_x, n_blocks_y = int(((n_cells_x - self.block_size[0]) / self.step_size) + 1), int(((n_cells_y - self.block_size[1]) / self.step_size) + 1)

        hog_cells = np.zeros((n_cells_x, n_cells_y, self.n_bins))
        for x in range(n_cells_x):
            for y in range(n_cells_y):
                m_block = gradient_magnitude[x * self.cell_size[0]: (x + 1) * self.cell_size[0], y * self.cell_size[1]: (y + 1) * self.cell_size[1]]
                g_block = gradient_angle[x * self.cell_size[0]: (x + 1) * self.cell_size[0], y * self.cell_size[1]: (y + 1) * self.cell_size[1]]
                hog_values_cell = self.compute_hog_cell(m_block, g_block)
                hog_cells[x, y] = hog_values_cell
        
        hog_descriptor = []
        for x in range(n_blocks_x):
            for y in range(n_blocks_y):
                block = hog_cells[x : x + self.block_size[0], y : y + self.block_size[1]]
                hog_descriptor += list(self.l2_normalize(block.ravel()))
        
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


    