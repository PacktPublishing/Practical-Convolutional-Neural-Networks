# import all required lib
import matplotlib.pyplot as plt
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')

inline
import numpy as np
from skimage.io import imread
from skimage.transform import resize

# Load a color image in grayscale
image = imread('sample_digit.png', as_grey=True)
image = resize(image, (28, 28), mode='reflect')
print('This image is: ', type(image),
      'with dimensions:', image.shape)
plt.imshow(image, cmap='gray')


def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[ x ][ y ], 2)), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[ x ][ y ] < thresh else 'black')


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
visualize_input(image, ax)
