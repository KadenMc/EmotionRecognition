import os, sys
import numpy as np
import cv2
import imutils

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

def visualize_img(img):
    cv2.namedWindow('visualize')
    cv2.moveWindow('visualize', 200, 200)
    cv2.resizeWindow("visualize", 500, 500)
    cv2.imshow('visualize', (cv2.resize(img, (0,0), fx=3, fy=3)).astype(np.uint8))
    cv2.waitKey(0)

def visualize_imgs(imgs, size, race=None):
    width = int(np.ceil(len(imgs) ** 0.5))
    plus_fifth = size + int(size/5)
    
    for i in range(len(imgs)):
        # Define window name, size & position 
        if race is not None:
            name = str(race[i]) + " - " + str(i)
        else:
            name = str(i)
        
        x = plus_fifth *(i % width) + size
        y = plus_fifth * np.floor(i//width) + size
        
        cv2.namedWindow(name)
        cv2.resizeWindow(name, size, size)
        cv2.moveWindow(name, int(x), int(y))
        
        # Show the image
        cv2.imshow(name, imutils.resize(imgs[i].astype(np.uint8), width=size))
    
    # Display images
    cv2.waitKey(0)
    