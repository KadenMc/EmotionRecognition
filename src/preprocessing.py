import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import pickle
import cv2

import sys

import os.path

from scipy.ndimage import generic_filter

from skimage.metrics import structural_similarity as compare_ssim

from sklearn.utils import shuffle

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


def shuffle_data(X, y):
    return shuffle(X, y)

def preprocess_X(X):
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


RACE_MAP = {
    -1: "undefined",
    1: "unknown",
    2: "white",
    3: "black",
    4: "asian",
    5: "brown",
    6: "invalid"
}

RACE_MAP_INV = {v: k for k, v in RACE_MAP.items()}


def get_similar_lists(inds):
    lsts = []
    sim_dict = dict()
    for ind in inds:
        if ind[0] in sim_dict:
            sim_dict[ind[1]] = sim_dict[ind[0]]
            lsts[sim_dict[ind[0]]].extend([ind[0], ind[1]])
        elif ind[1] in sim_dict:
            sim_dict[ind[0]] = sim_dict[ind[1]]
            lsts[sim_dict[ind[1]]].extend([ind[0], ind[1]])
        else:
            lsts.append([ind[0], ind[1]])
            sim_dict[ind[0]] = len(lsts) - 1
            sim_dict[ind[1]] = len(lsts) - 1

    lsts = [list(set(lst)) for lst in lsts]
    return lsts


def find_similar_images(X, save_every=100, search_complete=True):
    if os.path.isfile('../data/similarInds.pickle'):
        with open('../data/similarInds.pickle', 'rb') as handle:
            similar_inds = pickle.load(handle)
        last_i = max([i[0] for i in similar_inds])
        similar_inds = [i for i in similar_inds if i[0] != last_i]
    else:
        similar_inds = []
        last_i = 0

    # If the search isn't already complete
    if not search_complete:
        for i in range(last_i, len(X) - 1):
            # Save indices every so often in the event of crashes, etc.
            if i % save_every == 0:
                with open('../data/similarInds.pickle', 'wb') as handle:
                    pickle.dump(similar_inds, handle)
            
            for j in range(i + 1, len(X)):
                score = compare_ssim(X[i, :], X[j, :])
                if score > 0.6:
                    similar_inds.append([i, j, score])

        # Save when complete
        with open('../data/similarInds.pickle', 'wb') as handle:
            pickle.dump(similar_inds, handle)

    # Calculate lists of similar images
    lsts = get_similar_lists(similar_inds)
    with open('../data/similarLists.pickle', 'wb') as handle:
        pickle.dump(lsts, handle)
    
    # Now that similar images have been determined manually
    # choose which image(s) to keep from each similar group
    txt = open('D:\ML\Emotion Recognition\Keep indices.txt','r').readlines()
    remove = []
    for i, lst in enumerate(lsts):
        lst_X = [X[i] for i in lst]
        
        while True:
            try:
                try:
                    keep = txt[i][14:].replace("\n", "")
                except:
                    vis.visualize_imgs(lst_X, numbered=True)
                    keep = input("Keep indices: ").lower()
                
                if keep == "all":
                    keep = [i for i in range(len(lst))]
                elif keep == "none":
                    keep = []
                else:
                    keep = [int(s.strip()) for s in keep.split(",")]
                
                break
            except:
                print("Invalid response. Response must be 'all', 'none' or comma-separated numbers")
        
        remove.extend([lst[i] for i in range(len(lst)) if not i in keep])

    remove = np.array(remove)
    
    # Save & return remove array
    with open('../data/similarRemove.pickle', 'wb') as handle:
        pickle.dump(remove, handle)

    return remove

def filter_data(arrs, bools=None, inds=None):
    
    if inds is None:
        if bools is None:
            raise Exception("Either bools or inds must be defined.")
        # Get indices
        inds = np.argwhere(bools)[:, 0]
    
    # Filter based on indices
    for i, arr in enumerate(arrs):
        arrs[i] = arr[inds]

    return arrs


def get_races(X, y, race):
    race_data = dict()
    for k in RACE_MAP:
        race_data[RACE_MAP[k]] = filter_data([X, y], bools=(race == k))

    return race_data

def remove_elements(a, b):
    """
    Removes elements in array b from elements in array a.
    
    Parameters:
    a (numpy.ndarray): The list from which elements will be removed
    b (numpy.ndarray): The elements to remove
    """
    return np.setdiff1d(a, np.intersect1d(a,b))

def prepare_data(file, size=(48, 48), remove_similar=False):
    # Load the file
    df = pd.read_csv(file)

    # Define the data
    X = df['pixels'].values
    X = np.array([np.reshape(np.array(x.split(' ')), size) for x in X.ravel()], dtype=np.uint8)
    y = df['emotion'].values
    race = df['race'].values

    print("Originally:", len(X))
    n_org = len(X)
    n = n_org
    
    # Remove similar images
    if remove_similar:
        if not 'similar' in df.columns:
            # Calculate similar images - Warning: Takes a long time and requires manually labour!
            remove_inds = find_similar_images(X, search_complete=True)
            keep_inds = remove_elements(np.arange(len(df)), remove_inds)
            df["similar"] = 1
            df.loc[keep_inds, "similar"] = 0
            df.to_csv(file[:-4] + "_sim.csv", index=False)
            X, y, race = filter_data([X, y, race], inds=keep_inds)        
        else:
            # Filter out similar images if already recorded
            X, y, race = filter_data([X, y, race], bools=(df["similar"] == 0))
        
        print("{} similar - {}% of the dataset".format(n - len(X), np.round(((n_org - len(X))/n_org)*100, 2)))

    # Otherwise, simply remove duplicate images
    else:
        if not 'duplicate' in df.columns:
            # Record duplicates if not in CSV already    
            X, first_occurence_inds = np.unique(X, axis=0, return_index=True)
            df["duplicate"] = 1
            df.loc[first_occurence_inds, "duplicates"] = 0
            df.to_csv(file[:-4] + "_dup.csv", index=False)
            y, race = filter_data([y, race], inds=first_occurence_inds)
        else:
            # Filter out duplicates if already recorded
            X, y, race = filter_data([X, y, race], bools=(df["duplicate"] == 0))
    
        print("{} duplicates - {}% of the dataset".format(n - len(X), np.round(((n_org - len(X))/n_org)*100, 2)))
    
    # Remove invalid images
    n = len(X)
    X, y, race = filter_data([X, y, race], bools=(race != RACE_MAP_INV["invalid"]))
    print("{} invalid - {}% of the dataset".format(n - len(X), np.round(((n_org - len(X))/n_org)*100, 2)))
    
    # Remove images with no race label
    n = len(X)
    X, y, race = filter_data([X, y, race], bools=(race != RACE_MAP_INV["undefined"]))
    print("{} no race - {}% of the dataset".format(n - len(X), np.round(((n_org - len(X))/n_org)*100, 2)))
    
    print("Final amount:", len(X))
    
    # Pre-process X
    X = preprocess_X(X)
    
    return X, y, race