#-------------------------------------------------------------------
# @author 
# @copyright (C) 2018, 
# @doc
#
# @end
# Created : 17. Apr 2018 4:01 AM
#-------------------------------------------------------------------

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Sample():
    def __init__(self):
        print('Class Created')
        image = cv2.imread(os.path.join(os.path.dirname(__file__),'image.jpg'), 1)
        print(image.shape)
        image_flat = np.asarray(image.flatten())
        unique, counts = np.unique(image_flat, return_counts=True)

        x = unique
        y = counts

        # plt.plot(x, y, 'r+')  # 'r+', 'bo'
        # plt.plot(x, y, 'go--', linewidth=2, markersize=2)
        plt.bar(x, y)
        plt.title('Histogram of Image')
        plt.show()

if __name__ == '__main__':
    Sample()