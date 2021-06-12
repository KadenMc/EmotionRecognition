import os
import time

import numpy as np
import matplotlib.pyplot as plt

import visualize as vis
import preprocessing as pre


def load_numpy(file):
    return np.load(file)


def load_nii_file(nii_file):
    img = nib.load(nii_file)
    return img.get_fdata()


def load_tif_file(tif_file):
    return plt.imread(tif_file)


def extension_matches(file, extension):
    if file[-len(extension):] == extension:
        return True
    return False


def load_file(file):

    extension_dict = {
        '.npy': load_numpy,
        '.npz': load_numpy,
        '.tif': load_tif_file,
        '.nii': load_nii_file,
        '.nii.gz': load_nii_file,
    }

    for key in extension_dict.keys():
        if extension_matches(file, key):
            return extension_dict[key](file)

    raise Exception('This file type is not currently supported.')


def numpy_parallel_load(files, processes=4):
    '''
    Load data files in parallel.

    files: An ordered list of the files to load
    processes: The number of processes
    '''
    from multiprocessing import Pool
    p = Pool(processes)
    return p.map(load_file, files)


def get_files(path):
    '''
    Returns files in a directory.
    '''
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file



def data_generator(config, parallel=True, processes=4, data_dir=None):
    pass
