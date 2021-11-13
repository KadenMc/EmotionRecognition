from os.path import join
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR

# Import local files
from argparsing import MODELS_PATH
import dataloader as dl


def get_device(verbose=True):
    """
    Get the device on which to train.
    Use a GPU if possible, otherwise CPU.
    """
    print("torch.cuda.is_available()", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        if device.type == 'cuda':
            print("Using Device:", torch.cuda.get_device_name(0))
        else:
            print("Using Device:", device)
    
    return device


def exit_training():
    from argparsing import SRC_PATH
    try:
        f = open(join(SRC_PATH, 'exit.txt'), 'r')
        exit_training = int(f.read())
        f.close()
    except:
        # If we cannot find the file, or the contents are invalid
        # (i.e., not "0" or "1"), then do not exit training
        exit_training = 0
        
        # Reset the broken/invalid file
        f = open(join(SRC_PATH, 'exit.txt'), 'w')
        f.write("0")
        f.close()
    
    # If manually exiting, reset the file to have text "1"
    # so this doesn't have to be changed back manually
    if exit_training:
        f = open(join(SRC_PATH, 'exit.txt'), 'w')
        f.write("0")
        f.close()
    
    return exit_training


class EarlyStopping:
    """
    A class implementing early stopping based on patience.
    If we are 'patience' epochs without an improvement in our metric,
    then end the training. Improvement may be an increase or decrease
    depending on whether np.argmin (improvement = a derease), or
    np.argmax (improvement = an increase) is used as the mode.
    """
    def __init__(self, patience, mode=np.argmin):
        self.history = np.array([])
        self.patience = patience
        self.mode = mode
    
    def step(self, metric, verbose=True):
        self.history = np.append(self.history, metric)
        last_improved_ind = len(self.history) - (self.mode(self.history) + 1)
        if last_improved_ind > self.patience:
            if verbose:
                print("Stopped - No improvement in {} epochs".format(self.patience))
            return True
        return False


class Model(nn.Module):
    """
    Model definition and training functionality.
    """
    def __init__(self, n_outputs, args):

        super(Model, self).__init__()

        # Define the model
        self.model = nn.Sequential(
            # (None, 1, 48, 48)
            
            # Conv layer 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # (None, 32, 48, 48)
            nn.ReLU(),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 32, 24, 24)
            nn.Dropout2d(p=0.2),
            
            # Conv layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (None, 64, 24, 24)
            nn.ReLU(),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (None, 64, 12, 12)
            nn.Dropout2d(p=0.25),
            
            # Flatten
            nn.Flatten(),
            
            # Linear layer 1
            nn.Linear(64 * 12 * 12, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            # Linear layer 2
            nn.Linear(4096, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            
            # Output - Note: Cross entropy loss has built-in softmax
            nn.Linear(512, n_outputs)  
        )
        
        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Define optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        
        # Define learning rate scheduler (handles decay)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=args.lr_decay_gamma)
        
        self.early_stopping = EarlyStopping(args.patience)

    def summary(self, input_size):
        return summary(self.model, input_size)

    def forward(self, x):
        # Perform the forward pass
        return self.model(x)


    def get_loss(self, loader, device):
        """
        Gets a loss and accuracy given the current model and a dataloader
        from which to pull data.
        """
        accuracy = 0
        loss = 0
        for X_batch, y_batch in loader:
            # Send batch to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Add to loss and accuracy
            probs = self(X_batch)
            loss += self.loss_fn(probs, y_batch).item()
            top_p, top_class = probs.topk(1, dim=1)
            equals = top_class == y_batch.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        return loss/len(loader), accuracy/len(loader)


def train(train_loader, val_loader, model, device, args, tensorboard_path=None, model_save_path=None):
    """
    Train the model
    """
    # Setup Tensorboard
    if tensorboard_path is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(tensorboard_path)
    
    model.to(device)

    train_losses = []
    val_losses = []
    val_accuracy = []
    for e in range(args.epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {e}")
            batch_loss = 0
            for X_batch, y_batch in tepoch:
                # Send batch data to device
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Training pass
                model.optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = model.loss_fn(y_pred, y_batch)
                loss.backward()
                model.optimizer.step()

                batch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            
        # Log statistics
        train_losses.append(batch_loss/len(train_loader))
        val_loss, val_acc = model.get_loss(val_loader, device)
        val_losses.append(val_loss)
        val_accuracy.append(val_acc)
            
        # Print the stats if verbose
        if args.verbose:
            print("Epoch {} Train Loss: {}".format(e, np.round(train_losses[-1], 5)))
            print("Epoch {} Val Loss: {}".format(e, np.round(val_losses[-1], 5)))
            print("Epoch {} Val Acc: {}".format(e, np.round(val_accuracy[-1], 5)))
        
        # Write to TensorBoard
        if tensorboard_path is not None:
            writer.add_scalar("Train Loss", train_losses[-1], e)
            writer.add_scalar("Val Loss", val_losses[-1], e)
            writer.add_scalar("Val Acc", val_accuracy[-1], e)
        
        # Update the learning rate
        model.scheduler.step()
        
        # Save model on each new, lowest validation loss
        if model_save_path is not None and val_losses[-1] == min(val_losses):
            torch.save(model.model.state_dict(), model_save_path)
            
            if args.verbose:
                print("Saved model")
    
        # Check for early stopping
        if model.early_stopping.step(val_losses[-1], verbose=args.verbose):
            break
            
        # Check whether the training is to be stopped manually
        if exit_training():
            print("Manually exiting training")
            break
    
        if args.verbose:
            print()
    
    # Prepare and return the history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracy': val_accuracy,
    }
    return history