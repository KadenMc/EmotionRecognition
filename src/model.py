import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, n_outputs):

        super(Model, self).__init__()

        # Define our layers
        self.net = nn.Sequential(
            # Begin with a convolutional layer
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),

            nn.ReLu(),
            
            # First max pooling layer
            nn.MaxPool2d(2),
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # More convolutional layers
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),

            # Second max pooling layer
            nn.MaxPool2d(2),
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Finally, 2 fully connected (linear) layers
            nn.Linear(256, 100),
            nn.Linear(100, n_outputs))
        

    def forward(self, x):
        # Perform the forward pass
        return self.net(x)