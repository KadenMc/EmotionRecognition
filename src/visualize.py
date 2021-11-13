from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils

def save_image(save_path, key, suffix=None):
    """
    Saves an image to a given path.
    """
    if save_path is not None:
        if suffix is None:
            file = join(save_path, key) + '.png'
        else:
            file = join(save_path, key) + suffix + '.png'
        plt.savefig(file)


def plot_history(history, save_path=None, suffix=None, alpha=1):
    """
    Plots a model training history.
    """
    plt.rcParams["figure.figsize"] = (15, 5)
    
    for i, key in enumerate(history):
        plt.figure(i)
        plt.title(key)
        plt.plot(range(len(history[key])), history[key], alpha=alpha, color='black', lw=2)
        save_image(save_path, key, suffix=suffix)
        
        if save_path is None:
            plt.show()


def visualize_img(img):
    cv2.namedWindow('visualize')
    cv2.moveWindow('visualize', 200, 200)
    cv2.resizeWindow("visualize", 500, 500)
    cv2.imshow('visualize', (cv2.resize(img, (0,0), fx=3, fy=3)).astype(np.uint8))
    cv2.waitKey(0)

def visualize_imgs(imgs, size_multiplier=1, size_cap=400, display=True, save_as=None, numbered=False):
    """
    Visualizes a series of square images.
    
    imgs (list): A list of square NumPy arrays
    size (int): The desired size of each image displayed
    """
    
    
    # Define combined image
    width = int(np.ceil(len(imgs) ** 0.5))
    height = int(np.ceil(len(imgs)/width))
    
    # Define image size based on number of images
    size = int((800/width) * size_multiplier)
    size = size if size <= size_cap else size_cap
    
    img = np.zeros((height*size, width*size), dtype=np.uint8) + 255
    
    # Fill in combined image
    for i in range(len(imgs)):
        x = int(size *(i % width))
        y = int(size * np.floor(i//width))
        img[y: y + size, x: x + size] = imutils.resize(imgs[i].astype(np.uint8), width=size)
        
        if numbered:
            cv2.putText(img, str(i), (x + size//10, y + int(size*(1/4))), cv2.FONT_HERSHEY_SIMPLEX, size//130, 255, thickness=2)
            cv2.putText(img, str(i), (x + size//10, y + int(size*(19/20))), cv2.FONT_HERSHEY_SIMPLEX, size//130, 0, thickness=2)
    
    # Define window name, size & position
    cv2.namedWindow("img")
    cv2.moveWindow("img", size, size)
    
    # Display image
    if display:
        cv2.imshow("img", img)
        cv2.waitKey(0)
    
    # Save the image
    if save_as is not None:
        cv2.imwrite(save_as, img)