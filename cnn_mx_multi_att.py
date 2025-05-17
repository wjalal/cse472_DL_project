import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleAttention, self).__init__()
        # Convolutions with different kernel sizes for multi-scale feature extraction
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        # Combine multi-scale features
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Extract multi-scale features
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        # Aggregate features and apply sigmoid for attention
        attention = self.sigmoid(feat1 + feat2 + feat3)
        # Refine the input features with the attention map
        return x * attention

class AgePredictionCNN(nn.Module):
    def __init__(self, input_shape):
        super(AgePredictionCNN, self).__init__()

        # Define convolutional and pooling layers
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(10, 60), stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.att1 = MultiScaleAttention(1)  # Multi-scale attention after conv1

        self.conv2 = nn.Conv2d(1, 1, kernel_size=(5, 15), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.att2 = MultiScaleAttention(1)  # Multi-scale attention after conv2

        self.conv3 = nn.Conv2d(1, 1, kernel_size=(2, 6), stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.att3 = MultiScaleAttention(1)  # Multi-scale attention after conv3

        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = None  # Placeholder to be initialized dynamically
        self.fc1_bn = None  # Placeholder for batch normalization after fc1
        self.fc2 = nn.Linear(512, 128)
        self.fc2_bn = nn.LayerNorm(128)
        self.dropout = nn.Dropout(p=0.3)  # Dropout
        self.fc3 = nn.Linear(129, 1)  # Adding 1 for the `Sex` input

        self.relu = nn.ReLU()
        self.initialize_fc1(input_shape)

    def initialize_fc1(self, input_shape):
        # Create a sample input to pass through the convolutional layers
        sample_input = torch.zeros(1, *input_shape)
        x = self.conv1(sample_input)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.att1(x)  # Apply attention after first pooling
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.att2(x)  # Apply attention after second pooling
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.att3(x)  # Apply attention after third pooling
        flattened_size = x.numel()
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc1_bn = nn.LayerNorm(512)

    def forward(self, x, sex):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.att1(x)  # Apply attention

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.att2(x)  # Apply attention

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.att3(x)  # Apply attention

        x = self.flatten(x)

        if self.fc1 is None:
            raise ValueError("fc1 layer has not been initialized. Call `initialize_fc1` with the input shape.")

        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Concatenate `Sex` input
        x = torch.cat((x, sex.unsqueeze(1)), dim=1)
        x = self.fc3(x)

        return x
