import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(AttentionBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()  # Batch size, Channels, Height, Width
        y = self.global_avg_pool(x).view(b, c)  # Squeeze spatial dimensions
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)  # Reshape to match input dimensions
        return x * y  # Scale input by attention weights


class AgePredictionCNN(nn.Module):
    def __init__(self, input_shape):
        super(AgePredictionCNN, self).__init__()

        # Define convolutional layers with 1 channel
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(10, 60), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(1, 1, kernel_size=(5, 15), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(1, 1, kernel_size=(2, 6), stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Attention blocks
        self.attention1 = AttentionBlock(channels=1)
        self.attention2 = AttentionBlock(channels=1)
        self.attention3 = AttentionBlock(channels=1)

        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = None  # Placeholder to initialize dynamically
        self.fc1_bn = None  # Placeholder for batch normalization after fc1
        self.fc2 = nn.Linear(512, 128)
        self.fc2_bn = nn.LayerNorm(128)
        self.dropout = nn.Dropout(p=0.1)  # Dropout with 10% probability
        self.fc3 = nn.Linear(129, 1)  # Adding 1 for the `Sex` input

        self.relu = nn.ReLU()
        self.initialize_fc1(input_shape)

    def initialize_fc1(self, input_shape):
        sample_input = torch.zeros(1, *input_shape)
        x = self.conv1(sample_input)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        flattened_size = x.numel()
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc1_bn = nn.LayerNorm(512)

    def forward(self, x, sex):
        # First convolution block with attention
        x = self.conv1(x)
        x = self.attention1(x)  # Apply attention
        x = self.relu(x)
        x = self.pool1(x)

        # Second convolution block with attention
        x = self.conv2(x)
        x = self.attention2(x)  # Apply attention
        x = self.relu(x)
        x = self.pool2(x)

        # Third convolution block with attention
        x = self.conv3(x)
        x = self.attention3(x)  # Apply attention
        x = self.relu(x)
        x = self.pool3(x)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)
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
