import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetworkWithCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim

        # For 4x4 board, reshape 16 input features to (1, 4, 4)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=1),  # -> (32, 3, 3)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),  # -> (64, 2, 2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
        )

        # Flattened CNN output will be 64 * 2 * 2 = 256
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # normalize: log2(tile_val) if > 0, else keep as 0 for stability
        # TODO: this might be a problematic line, but essentially it's applying log2 
        # on tiles since they are all exponents of 2. I think we should keep it
        # for stability reasons
        x = torch.where(x > 0, torch.log2(x), torch.zeros_like(x))

        # x shape: (batch_size, 16)
        batch_size = x.size(0)

        # Reshape to
        x = x.view(batch_size, 1, 4, 4)
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)  # Flatten to (batch_size, 256)
        return self.fc_layers(x)
