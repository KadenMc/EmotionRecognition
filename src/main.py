import os
import argparse
import yaml
import pickle

import numpy as np
import torch
import torch.nn as nn

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
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('train', type=path, help='Training data file')
    parser.add_argument('--config', type=file_path, default=__location__ + '/config.yaml', \
        help='YAML config file specifying data paths, training hyperparameters and prediction settings')
    parser.add_argument('--load_weights', type=file_path, help='Load model weights from file')

    parser.add_argument("--remove_similar", action='store_true', \
        help="Find and remove similar data if true. Must calculate similar images (intensive) if not already saved.")

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


def main():

    # Parse the arguments
    args = parse_arguments()

    # Load the data from CSV, process, and pickle
    if args.train.endswith('.csv'):
        # Process data
        X, y, race = pre.prepare_data(args.train, remove_similar=args.remove_similar)
        
        # Save all data
        with open('../data/Xy.pickle', 'wb') as handle:
            pickle.dump((X, y), handle)
        
        # Save race-specific data
        race_data = pre.get_races(X, y, race)
        min_race_length = min([len(race_data[k][0]) for k in race_data if \
            len(race_data[k][0]) != 0 and k not in ["unknown"]])
        Xs_equal = []
        ys_equal = []
        for key in race_data:
            if len(race_data[key][0]) != 0:
                print(key, ": ", np.round((len(race_data[key][0])/len(X))*100, 2), "%")
        
                # Extract and shuffle race data
                X_race, y_race = race_data[key]
                X_race, y_race = pre.shuffle_data(X_race, y_race)
                
                # Save a number of random samples equal to the minimum length race dataset
                equal_inds = np.random.choice(len(X_race), size=min_race_length, replace=False)
                Xs_equal.append(X_race[equal_inds])
                ys_equal.append(y_race[equal_inds])
                
                # Visualize random samples from each race
                display_inds = np.random.choice(len(X_race), size=16, replace=False)
                vis.visualize_imgs(X_race[display_inds]*255)
                
                # Save as pickled data
                with open('../data/Xy_{}.pickle'.format(key), 'wb') as handle:
                    pickle.dump((X_race, y_race), handle)

        # Save equal amount of data from each race (not including unknown)
        X_equal = np.concatenate(Xs_equal)
        y_equal = np.concatenate(ys_equal)
        X_equal, y_equal = pre.shuffle_data(X_equal, y_equal)
        print("Min race dataset size:", min_race_length)
        print("Equal dataset size", len(X_equal))
        
        with open('../data/Xy_equal.pickle', 'wb') as handle:
            pickle.dump((X_equal, y_equal), handle)
    
    # Train
    else:
        # Load from pickled data
        with open(args.train, 'rb') as handle:
            X, y = pickle.load(handle)

        # Load the config
        config = load_config(args.config)

        # Train
        # # of outputs = # of unique y values
        num_outputs = len(np.unique(y))

        m = model.Model(num_outputs)

        # Test model with a random input
        #import torch
        #input = torch.randn(1, 1, 32, 32)
        #out = m(input)
        #print(out)

        # TODO:
        # Define device
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
