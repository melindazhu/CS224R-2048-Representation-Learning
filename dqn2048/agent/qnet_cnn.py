import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetworkWithCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = 128

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

        self.shared_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Auxiliary tasks
        # 1. Action legality prediction; reduce illegal moves made by the agent
        self.q_head = nn.Linear(128, output_dim)
        self.legal_head = nn.Linear(128, output_dim)

        # 2. Future max tile prediction; train the agent to predict the max
        # tile it will reach by the end of the episode => learn representations that 
        # are predictive of long-term tile growth
        self.max_tile_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # regression output: predicted log2(max_tile)
        )

        # # Flattened CNN output will be 64 * 2 * 2 = 256
        # NOTE: used w/o auxiliary tasks
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(128, output_dim)
        # )

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
        # return self.fc_layers(x)
        x = self.shared_fc(x) # uncomment the above if not using auxiliary tasks

        # Auxiliary tasks: comment everything below if not using auxiliary tasks
        q_values = self.q_head(x)
        legal_logits = self.legal_head(x)
        max_tile_pred = self.max_tile_head(x)

        return q_values, legal_logits, max_tile_pred
    