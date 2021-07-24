import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import generic_filter

import visualize as vis


def convolution(input, kernel):
    return as_strided(
    input,
    shape=(
        input.shape[0] - kernel.shape[0] + 1,  # The feature map is a few pixels smaller than the input
        input.shape[1] - kernel.shape[1] + 1,
        kernel.shape[0],
        kernel.shape[1],
    ),
    strides=(
        input.strides[0],
        input.strides[1],
        input.strides[0],  # When we move one step in the 3rd dimension, we should move one step in the original data too
        input.strides[1],
    ),
    writeable=False,  # totally use this to avoid writing to memory in weird places
)

def normalize_intensity(x, _):
    return (x - x.mean()) / (6 * np.std(x))


def local_intensity_normalization(x, k):
    c = convolution(x, np.ones((k,k)))
    c = c[::k, ::k]
    c = np.apply_over_axes(normalize_intensity, c, [0, 1])
    return c.reshape(x.shape)

def local_contrast_normalization(x):
    '''
    The value of all pixels is subtracted from a Gaussian-weighted average of its neighbors.
    Later, each pixel is divided by the standard deviation of its own neighbor pixels.
    '''
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    x = x.astype(np.float)
    neigbhour_mean = generic_filter(x, np.mean, size=3)#footprint=kernel)
    neigbhour_std = generic_filter(x, np.std, size=3)#footprint=kernel)
    return (x - neigbhour_mean)/neigbhour_std



def preprocess(X):
    """
    Preprocess input X.

    x: Data to process
    """

    X = X.astype(float)

    '''
    for ind in range(X.shape[0]):

        # Local intensity normalization
        X[ind] = local_intensity_normalization(X[ind].astype(np.float), 4)

        # Local contrast normalization
        X[ind] = local_contrast_normalization(X[ind])

        #print(X[ind])
        #vis.visualize_image(X[ind])
        
    #n = cv2.equalizeHist(X[ind])
    '''
    X = X/255
    return X
