import os, sys
import numpy as np


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
