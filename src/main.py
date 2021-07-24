import os
import argparse
import yaml
import pickle

import numpy as np
import torch
import torch.nn as nn

import dataLoader as dL
import visualize as vis
import preprocessing as pre
import model


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
    parser.add_argument('train', type=path, help='Training data file')
    parser.add_argument('config', type=file_path, default='config.yaml', \
        help='YAML config file specifying data paths, training hyperparameters and prediction settings')
    parser.add_argument('--load_weights', type=file_path, help='Load model weights from file')
    #parser.add_argument('--save_weights', default=os.path.join(MODEL_PATH, 'model.h5'), \
    #    help='Save model weights to file')
    #parser.add_argument('--history', default=os.path.join(VISUALIZE_PATH, 'history.png'), \
    #    help='Path to save model history')
    #parser.add_argument("--verbose", type=int, default=2, help="Training verbosity")
    #parser.add_argument("--predict", action='store_true', \
    #    help="Predict the data if true, otherwise train by default")
    #parser.add_argument("--not_parallel", action='store_true', \
    #    help="If flagged, do not load the data in parallel")
    #parser.add_argument("--processes", type=int, default=4, \
    #    help="The number of processes for loading data in parallel")
    
    args = parser.parse_args()
    return args


def load_config(file):
    try: 
        with open (file, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        print('Error reading the config file.')


def dataloader():
    # See torch.utils.data.DataLoader as an example

    #X= Variable(torch.from_numpy(X), requires_grad=False)
    #X= Variable(torch.from_numpy(X).float(), requires_grad=False)
    pass

def main():

    # Parse the arguments
    args = parse_arguments()

    # Load the data from CSV (and pickle)
    if args.train.endswith('.csv'):
        # Process CSV
        X, y = dL.load_csv_data(args.train)

        # Save as pickled data
        with open('../data/Xy.pickle', 'wb') as handle:
            pickle.dump((X,y), handle)
    else:
        # Load directly from pickled data
        with open(args.train, 'rb') as handle:
            X, y = pickle.load(handle)


        # Preprocess & visualize
        X = pre.preprocess(X)
        vis.visualize_image(X[0]*255)
        #vis.visualize_image(X[1])
        #vis.visualize_image(X[2])

    # Load the config
    config = load_config(args.config)

    exit()

    # # of outputs = # of unique y values
    num_outputs = len(np.unique(y))

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
    epochs = config["training"]["epochs"]

    optimizer = torch.optim.Adam(m.parameters(), lr=4e-2, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        #avg_cost = 0
        #total_batch = len(dataloader)
        
        #for x, y in dataloader:
        #    x = x.to(device)
        #    y = y.to(device)

        # Forward pass
        y_pred = model(X)

        # Compute loss
        loss = loss_fn(y_pred, y)
        print("Epoch {} loss: {}".format(epoch, loss))

        # Compute gradient/backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()


if __name__ == '__main__':
    main()
