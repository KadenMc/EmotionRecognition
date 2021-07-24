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
    cv2.namedWindow('visualize')
    cv2.moveWindow('visualize', 200, 200)
    cv2.resizeWindow("visualize", 500, 500)
    cv2.imshow('visualize', (cv2.resize(arr, (0,0), fx=3, fy=3)).astype(np.uint8))
    cv2.waitKey(0)