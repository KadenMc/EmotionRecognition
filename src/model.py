import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n_outputs):

        super(Model, self).__init__()

        # Define our layers
        self.net = nn.Sequential(
            # Begin with a convolutional layer
            nn.Conv2d(64, kernel_size=5, stride=2),

            # First max pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2),

            # More convolutional layers
            nn.Conv2d(32, kernel_size=3, stride=1),
            nn.Conv2d(32, kernel_size=3, stride=1),

            # Second max pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, kernel_size=3, stride=1),#, bias=False),

            # Finally, 2 fully connected (linear) layers with ReLu/dropout
            nn.Linear(1024, 512),
            nn.ReLu(),
            nn.Dropout(0.3),

            nn.Linear(512, n_outputs),
            nn.ReLu(),
            nn.Dropout(0.3),

            # Softmax activation as final output
            nn.Softmax())

    def forward(self, x):
        # Perform the forward pass
        return self.net(x)
