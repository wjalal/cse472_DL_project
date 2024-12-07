import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        # Define key, query, and value transformations
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)  # For attention scores

    def forward(self, x):
        # Compute Q, K, V matrices
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)

        # Compute weighted value representation
        weighted_values = torch.matmul(attention_weights, V)

        return weighted_values


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

        # Attention layer
        self.attention = None

        self.flatten = nn.Flatten()

        # Fully connected layers (fc1 dimensions are calculated dynamically)
        self.fc1 = None  # Placeholder to be initialized dynamically
        self.fc1_bn = None  # Placeholder for batch normalization after fc1
        self.fc2 = nn.Linear(512, 128)
        self.fc2_bn = nn.LayerNorm(128)
        self.dropout = nn.Dropout(p=0.1)  # Dropout with 10% probability
        self.fc3 = nn.Linear(129, 1)  # Adding 1 for the `Sex` input

        self.relu = nn.ReLU()
        self.initialize_fc1(input_shape)

    def initialize_fc1(self, input_shape):
        # Create a sample input to pass through the attention and convolutional layers
        sample_input = torch.zeros(1, *input_shape)

        x = self.conv1(sample_input)
        x = self.relu(x)
        x = self.pool1(x)  # Apply pooling
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)  # Apply pooling
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)  # Apply pooling

        # Process through attention layer (requires reshaping to match attention input format)
        x = x.squeeze(1)  # Remove channel dim for attention
        # Attention layer
        self.attention = SelfAttention(input_dim=x.shape[-1])  # Assuming last dim is the feature size
        x = self.attention(x)
        x = x.unsqueeze(1)  # Restore channel dim

        flattened_size = x.numel()  # Total number of elements after flattening
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc1_bn = nn.LayerNorm(512)  # Initialize batch normalization for fc1

    def forward(self, x, sex):
        # Pass through convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)  # Apply pooling
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)  # Apply pooling
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)  # Apply pooling

        # Apply attention layer
        x = x.squeeze(1)  # Remove channel dimension for attention
        x = self.attention(x)
        x = x.unsqueeze(1)  # Restore channel dimension

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
