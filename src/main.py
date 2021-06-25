import os
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn

import dataLoader as dL
import visualize as vis
import model

# Define relative paths
CUR_PATH = str(os.path.dirname(os.path.abspath(__file__)))
ROOT_PATH = CUR_PATH[:CUR_PATH.rindex('/')]
VISUALIZE_PATH = os.path.join(ROOT_PATH, '/visualizations/')
MODEL_PATH = os.path.join(ROOT_PATH, '/model/')
DATA_PATH = os.path.join(ROOT_PATH, '/data/')


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


def parse_arguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument('train_path', type=path, help='Data file or folder path')
    parser.add_argument('--config', type=file_path, default='config.yaml', \
        help='YAML config file specifying data paths, training hyperparameters and prediction settings')
    parser.add_argument('--load_weights', type=file_path, help='Load model weights from file')
    parser.add_argument('--save_weights', default=os.path.join(MODEL_PATH, 'model.h5'), \
        help='Save model weights to file')
    parser.add_argument('--history', default=os.path.join(VISUALIZE_PATH, 'history.png'), \
        help='Path to save model history')
    parser.add_argument("--verbose", type=int, default=2, help="Training verbosity")
    parser.add_argument("--predict", action='store_true', \
        help="Predict the data if true, otherwise train by default")
    parser.add_argument("--not_parallel", action='store_true', \
        help="If flagged, do not load the data in parallel")
    parser.add_argument("--processes", type=int, default=4, \
        help="The number of processes for loading data in parallel")
    args = parser.parse_args()
    return args


def load_config(file):
    default_config = dict()
    training = {
        'batch_size': 32,
        'epochs': 20,
        'steps_per_epoch': 5,
        
        'validation_percent': 0.15,
        'test_percent': 0.15,
    }
    default_config['training'] = training

    if file is not None:
        try: 
            with open (file, 'r') as file:
                config = yaml.safe_load(file)
                return config
        except Exception as e:
            print('Error reading the config file. Using default config.')

    return default_config


def dataloader():
    # See torch.utils.data.DataLoader as an example

    #X= Variable(torch.from_numpy(X), requires_grad=False)
    #X= Variable(torch.from_numpy(X).float(), requires_grad=False)
    pass

def main():

    # Parse the arguments
    args = parse_arguments()

    # Load the YAML config
    config = load_config(args.config)

    # Create the data generator
    #generator = dL.data_generator(config, parallel=(not args.not_parallel), processes=args.processes, data_dir=DATA_PATH)

    num_outputs = 8

    m = model.Model(num_outputs)

    # Test model with a random input
    #import torch
    #input = torch.randn(1, 1, 32, 32)
    #out = m(input)
    #print(out)

    # TODO:
    # Create dataloader
    # Define device
    dataloader = None
    device = None

    # Load from config
    epochs = 100

    optimizer = torch.optim.Adam(m.parameters(), lr=4e-2, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        avg_cost = 0
        total_batch = len(dataloader)
        
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            y_pred = model(x)

            # Compute loss
            loss = loss_fn(y_pred, y)

            # Compute gradient/backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update parameters
            optimizer.step()


if __name__ == '__main__':
    main()
