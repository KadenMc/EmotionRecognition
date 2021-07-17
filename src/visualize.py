import os, sys
import numpy as np
import cv2

def plot_history(hist, loss_name='loss', path=None):
    """
    Plot the training history of a model.

    hist: The training history
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    if path is None:
        plt.show()
    else:
        plt.savefig(path)

def visualize_image(arr):
    print(arr.shape)
    cv2.namedWindow('test')
    cv2.moveWindow('test', 200, 200)
    cv2.imshow('test', (arr*255).astype(np.uint8))
    cv2.waitKey(0)