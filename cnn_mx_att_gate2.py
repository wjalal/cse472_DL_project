import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        # Resize g to match x's spatial dimensions
        g = nn.functional.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi


class AgePredictionCNN(nn.Module):
    def __init__(self, input_shape):
        super(AgePredictionCNN, self).__init__()

        # Define convolutional and pooling layers
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(10, 60), stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv2 = nn.Conv2d(1, 1, kernel_size=(5, 15), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv3 = nn.Conv2d(1, 1, kernel_size=(2, 6), stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)

        # Attention Gates for spatial refinement
        self.ag1 = AttentionGate(F_g=1, F_l=1, F_int=1)
        self.ag2 = AttentionGate(F_g=1, F_l=1, F_int=1)

        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = None  # Placeholder to be initialized dynamically
        self.fc1_bn = None  # Placeholder for batch normalization after fc1
        self.fc2 = nn.Linear(512, 128)
        self.fc2_bn = nn.LayerNorm(128)
        self.dropout = nn.Dropout(p=0.3)  # Dropout with 30% probability
        self.fc3 = nn.Linear(129, 1)  # Adding 1 for the `Sex` input

        self.relu = nn.ReLU()
        self.initialize_fc1(input_shape)

    def initialize_fc1(self, input_shape):
        # Create a sample input to pass through the convolutional layers
        sample_input = torch.zeros(1, *input_shape)
        x1 = self.conv1(sample_input)
        x1 = self.relu(x1)
        x1 = self.pool1(x1)

        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x2 = self.pool2(x2)

        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        x3 = self.pool3(x3)

        # Attention Gates
        x2 = self.ag1(g=x3, x=x2)  # First attention gate
        x1 = self.ag2(g=x2, x=x1)  # Second attention gate

        flattened_size = x1.numel()  # Total number of elements after flattening
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc1_bn = nn.LayerNorm(512)  # Initialize batch normalization for fc1

    def forward(self, x, sex):
        # Initial convolution layers
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.pool1(x1)

        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x2 = self.pool2(x2)

        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        x3 = self.pool3(x3)

        # Attention Gates
        x2 = self.ag1(g=x3, x=x2)  # First attention gate
        x1 = self.ag2(g=x2, x=x1)  # Second attention gate

        # Flatten and fully connected layers
        x = self.flatten(x1)

        if self.fc1 is None:
            raise ValueError("fc1 layer has not been initialized. Call `initialize_fc1` with the input shape.")

        x = self.fc1(x)
        x = self.fc1_bn(x)  # Apply batch normalization
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout

        x = self.fc2(x)
        x = self.fc2_bn(x)  # Apply batch normalization
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout

        # Concatenate `Sex` input
        x = torch.cat((x, sex.unsqueeze(1)), dim=1)
        x = self.fc3(x)

        return x
