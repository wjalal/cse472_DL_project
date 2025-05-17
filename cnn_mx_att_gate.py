import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Ensure kernel size is odd for consistent padding
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average and Max pooling across the channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along channel dimension
        pooled = avg_pool + max_pool
        # Apply convolution and sigmoid activation
        attention_map = self.sigmoid(self.conv(pooled))
        # Multiply input with attention map
        return x * attention_map


class AgePredictionCNN(nn.Module):
    def __init__(self, input_shape):
        super(AgePredictionCNN, self).__init__()

        # Define convolutional and pooling layers
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(10, 60), stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.sa1 = SpatialAttention(kernel_size=7)  # Add spatial attention after first layer

        self.conv2 = nn.Conv2d(1, 1, kernel_size=(5, 15), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.sa2 = SpatialAttention(kernel_size=5)  # Add spatial attention after second layer

        self.conv3 = nn.Conv2d(1, 1, kernel_size=(2, 6), stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.sa3 = SpatialAttention(kernel_size=3)  # Add spatial attention after third layer

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
        x = self.conv1(sample_input)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.sa1(x)  # Apply spatial attention
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.sa2(x)  # Apply spatial attention
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.sa3(x)  # Apply spatial attention
        flattened_size = x.numel()  # Total number of elements after flattening
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc1_bn = nn.LayerNorm(512)  # Initialize batch normalization for fc1

    def forward(self, x, sex):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.sa1(x)  # Apply spatial attention

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.sa2(x)  # Apply spatial attention

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.sa3(x)  # Apply spatial attention

        x = self.flatten(x)

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
