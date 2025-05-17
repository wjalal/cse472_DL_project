# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def conv_block(in_ch, out_ch, kernel_size, padding, groups):
#     return nn.Sequential(
#         nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
#         nn.GroupNorm(num_groups=groups, num_channels=out_ch),
#         nn.SiLU(),
#         nn.MaxPool2d(kernel_size=2, stride=1),
#         nn.Dropout2d(p=0.1),
#     )

# class AgePredictionCNN(nn.Module):
#     def __init__(self,input_shape=None):
#         super(AgePredictionCNN, self).__init__()

#         # Convolutional blocks with GroupNorm
#         self.block1 = conv_block(1, 16, kernel_size=(10, 60), padding=(5, 30), groups=4)
#         self.block2 = conv_block(16, 32, kernel_size=(5, 15), padding=(2, 7), groups=8)
#         self.block3 = conv_block(32, 64, kernel_size=(2, 6), padding=(1, 3), groups=16)

#         # Adaptive pooling to get fixed-size output regardless of input dims
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

#         # Fully connected layers
#         self.fc1 = nn.Linear(64, 128)
#         self.fc1_gn = nn.GroupNorm(num_groups=8, num_channels=128)
#         self.dropout = nn.Dropout(p=0.2)

#         # Final layer: +1 for sex input
#         self.fc2 = nn.Linear(128 + 1, 1)

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.Linear)):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x, sex):
#         # x: [B, 1, H, W], sex: [B]
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)

#         x = self.global_pool(x)      # Shape: [B, 64, 1, 1]
#         x = torch.flatten(x, 1)      # Shape: [B, 64]

#         x = self.fc1(x)
#         x = self.fc1_gn(x)
#         x = F.silu(x)
#         x = self.dropout(x)

#         x = torch.cat([x, sex.unsqueeze(1)], dim=1)  # Concatenate sex
#         out = self.fc2(x)

#         return out

# import torch
# import torch.nn as nn
# import torch.optim as optim

# class AgePredictionCNN(nn.Module):
#     def __init__(self, input_shape):
#         super(AgePredictionCNN, self).__init__()
#         # Define convolutional and pooling layers with typical kernel sizes and GroupNorm
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=1)
#         self.gn1 = nn.GroupNorm(num_groups=4, num_channels=16)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1)
#         self.gn2 = nn.GroupNorm(num_groups=4, num_channels=32)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
#         self.gn3 = nn.GroupNorm(num_groups=8, num_channels=64)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # Adaptive pooling to get fixed-size output regardless of input dims
#         self.flatten = nn.Flatten()

#         # Fully connected layers
#         self.fc1 = None  # Placeholder to be initialized dynamically
#         self.fc1_ln = None  # LayerNorm after fc1
#         self.fc2 = nn.Linear(512, 128)
#         self.fc2_ln = nn.LayerNorm(128)
#         self.dropout = nn.Dropout(p=0.4)
#         self.fc3 = nn.Linear(129, 1)  # Adding 1 for the `Sex` input

#         self.swish = nn.SiLU()

#         self.initialize_fc1(input_shape)
#         self.initialize_weights()

#     def initialize_fc1(self, input_shape):
#         # Create a sample input to pass through the convolutional layers
#         sample_input = torch.zeros(1, *input_shape)
#         x = self.conv1(sample_input)
#         x = self.swish(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.swish(x)
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = self.swish(x)
#         x = self.pool3(x)
#         flattened_size = x.numel()
#         self.fc1 = nn.Linear(flattened_size, 512)
#         self.fc1_ln = nn.LayerNorm(512)

#     def initialize_weights(self):
#         # Initialize weights for all layers
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x, sex):
#         x = self.conv1(x)
#         x = self.gn1(x)
#         x = self.swish(x)
#         x = self.pool1(x)

#         x = self.conv2(x)
#         x = self.gn2(x)
#         x = self.swish(x)
#         x = self.pool2(x)

#         x = self.conv3(x)
#         x = self.gn3(x)
#         x = self.swish(x)
#         x = self.pool3(x)

#         x = self.flatten(x)

#         if self.fc1 is None:
#             raise ValueError("fc1 layer has not been initialized. Call `initialize_fc1` with the input shape.")

#         x = self.fc1(x)
#         x = self.fc1_ln(x)
#         x = self.swish(x)
#         x = self.dropout(x)

#         x = self.fc2(x)
#         x = self.fc2_ln(x)
#         x = self.swish(x)
#         x = self.dropout(x)

#         x = torch.cat((x, sex.unsqueeze(1)), dim=1)
#         x = self.fc3(x)

#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class AgePredictionCNN(nn.Module):
    def __init__(self, input_shape):
        super(AgePredictionCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2d1 = nn.Dropout2d(p=0.3)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2d2 = nn.Dropout2d(p=0.3)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=1)
        self.gn3 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2d3 = nn.Dropout2d(p=0.3)

        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = None
        self.fc1_ln = None
        self.fc2 = nn.Linear(256, 128)
        self.fc2_ln = nn.LayerNorm(128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(129, 1)  # 128 + 1 (sex)

        self.swish = nn.SiLU()

        self.initialize_fc1(input_shape)
        self.initialize_weights()

    def initialize_fc1(self, input_shape):
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            x = self.forward_conv(sample_input)
            flattened_size = x.numel()
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc1_ln = nn.LayerNorm(256)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward_conv(self, x):
        x = self.pool1(self.swish(self.gn1(self.conv1(x))))
        x = self.drop2d1(x)

        x = self.pool2(self.swish(self.gn2(self.conv2(x))))
        x = self.drop2d2(x)

        x = self.pool3(self.swish(self.gn3(self.conv3(x))))
        x = self.drop2d3(x)

        return self.flatten(x)

    def forward(self, x, sex):
        x = self.forward_conv(x)

        x = self.swish(self.fc1_ln(self.fc1(x)))
        x = self.dropout(x)

        x = self.swish(self.fc2_ln(self.fc2(x)))
        x = self.dropout(x)

        x = torch.cat((x, sex.unsqueeze(1)), dim=1)
        x = self.fc3(x)

        return x
