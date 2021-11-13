import re
import sys
import os
from os.path import join
import pickle
import numpy as np
from torchvision import transforms

# Import local files
import argparsing as ap
import model as m
import visualize as vis


def stdout_to_file_setup(args):
    """
    Sends stdout output to file instead of displaying it on-screen.
    Helpful for keeping track of hyperparameters and model performance.
    """
    
    # Find a log number - smallest possible which is 0 and above
    logs = sorted([f for f in os.listdir(ap.OUTPUT_PATH) if 'log' in f and \
        os.path.isfile(join(ap.OUTPUT_PATH, f))])
    if len(logs) == 0:
        num = 0
    else:
        logs = [int(re.findall(r'\d+', l)[0]) for l in logs]
        logs.sort()
        found = False
        for i in range(len(logs)):
            if i != logs[i]:
                found = True
                num = i
                break
        
        if not found:
            num = len(logs)
    
    stdout_origin=sys.stdout 
    sys.stdout = open(join(ap.OUTPUT_PATH, "log{}.txt".format(num)), "w")
    suffix = str(num)
    return stdout_origin, suffix


def get_tensorboard_path(args, suffix):
    if args.use_tensorboard:
        if suffix is not None:
            path = join(ap.OUTPUT_PATH, 'log{}'.format(suffix))
        else:
            path = join(ap.OUTPUT_PATH, 'log')
    else:
        path = None
    return path


def get_model_save_path(suffix):
    if suffix is not None:
        path = join(ap.MODELS_PATH, 'model{}'.format(suffix))
    else:
        path = join(ap.MODELS_PATH, 'model')
    return path + '.pt'


def load_pickled(args):
    # Load from pickled data
    with open(args.data, 'rb') as handle:
        X, y = pickle.load(handle)
        X = np.expand_dims(X.astype(np.float32), axis=1)
        y = y.astype(np.float32)
        return X, y


def main():
    # Parse the arguments
    args = ap.main_parse_arguments()

    # Assert that the file being loaded isn't a CSV file
    if args.data.endswith('.csv'):
        print("Please preprocess the CSV first. See README for details.")
        exit()
    
    # Setup logging stdout to file
    if args.stdout_to_file:
        stdout_origin, suffix = stdout_to_file_setup(args)
    else:
        suffix = None

    # Load and prepare the data
    X, y = load_pickled(args)

    print("Data shape:", X.shape)
    print("Targets shape:", y.shape)

    # Put data into PyTorch DataLoaders
    transform = transforms.Compose([
        #transforms.Grayscale(), # Convert to grayscale
        #transforms.ToTensor(), # Make into PyTorch tensor
        transforms.Normalize((0.5,), (0.5,)), # Normalization
    ])
    from dataloader import prepare_dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(X, y, args, transform=transform)

    # Define device - Use GPU if possible
    device = m.get_device()

    # Define the model
    n_outputs = len(np.unique(y))
    print("n_outputs", n_outputs)
    model = m.Model(n_outputs, args)
    model.to(device)

    # Predict    
    if args.predict:
        assert args.model_path is not None
        y = test_loader.dataset.dataset.targets
        y_pred = model.infer(test_loader, device)
        
        loss = model.loss_fn(y_pred, y).item()
        top_p, top_class = y_pred.topk(1, dim=1)
        equals = top_class == y.view(*top_class.shape)
        
        import torch
        accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
        
        print("Test loss:", loss)
        print("Test accuracy:", accuracy)
        
        y = y.detach().numpy()
        y_pred = y_pred.detach().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        cf_matrix = confusion_matrix(y, y_pred)#, labels=["ant", "bird", "cat"])
        sns.heatmap(cf_matrix, annot=True)
        plt.show()
    
    # Train
    else:
        
        tensorboard_path = get_tensorboard_path(args, suffix)
        model_save_path = get_model_save_path(suffix)
        
        # Train the model
        history = m.train(train_loader, val_loader, model, device, args, \
            tensorboard_path=tensorboard_path, model_save_path=model_save_path)

        vis.plot_history(history)

    # Close the stdout pipe if stdout was being sent to a file
    if args.stdout_to_file:
        sys.stdout.close()
        sys.stdout=stdout_origin


if __name__ == '__main__':
    main()
