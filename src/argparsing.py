import argparse
import os
from os.path import join


# Define relative paths
SRC_PATH = str(os.path.dirname(os.path.abspath(__file__)))
try:
    ROOT_PATH = SRC_PATH[:SRC_PATH.rindex('/')]
except:
    ROOT_PATH = SRC_PATH[:SRC_PATH.rindex('\\')]
DATA_PATH = os.path.join(ROOT_PATH, 'data')
OUTPUT_PATH = os.path.join(ROOT_PATH, 'outputs')
MODELS_PATH = os.path.join(ROOT_PATH, 'models')

# Define argparse types
def path(path):
    if os.path.isdir(path) or os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not a valid path".format(path))


def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not a valid file path".format(path))


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not a valid directory path".format(path))


def percent(p):
    if p < 0:
        return argparse.ArgumentTypeError("{} is not a valid percentage".format(p))
    # Check whether a decimal percentage was inputed
    if p > 1:
        if p > 100:
            return argparse.ArgumentTypeError("{} is not a valid percentage".format(p))
        return p/100
    return p


# Argparsing functions
def main_parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument('data', type=path, help='Data file')
    
    # Training arguments
    parser.add_argument("--val_percent", type=percent, default=0.15, help="Validation dataset percentage")
    parser.add_argument("--test_percent", type=percent, default=0.15, help="Test dataset percentage")
    parser.add_argument("--lr", type=float, default=0.001, help="Number of epochs")
    parser.add_argument("--lr_decay_gamma", type=float, default=1, help="StepLR decay gamma. Default is 1 - no decay")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("-e", "--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Training batch size")
    
    # Logging arguments
    parser.add_argument("--stdout_to_file", action='store_true', \
        help="If flagged, log standard output to file in output path")
    parser.add_argument("-v", "--verbose", action='store_true', \
        help="Specify training verbosity")
    parser.add_argument("--use_tensorboard", action='store_true', \
        help="Specify whether to use TensorBoard")
    
    # Predict arguments
    parser.add_argument("--predict", action='store_true', \
        help="If flagged, predict, otherwise train")
    parser.add_argument('--model_path', type=file_path, \
        help='Path from which to load model parameters. Must be specified if --predict flagged.')
    
    #parser.add_argument('--history', default=os.path.join(VISUALIZE_PATH, 'history.png'), \
    #    help='Path to save model history')
    #parser.add_argument("--predict", action='store_true', \
    #    help="Predict the data if true, otherwise train by default")
    
    args = parser.parse_args()
    return args


def preprocessing_parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=path, help='CSV file')
    parser.add_argument("--remove_similar", action='store_true', \
        help="Find and remove similar data if true. Must calculate similar images (intensive) if not already saved.")
    parser.add_argument("--visualize", action='store_true', \
        help="Visualize the different processed groups.")
    
    args = parser.parse_args()
    return args