import torch
import torch.nn as nn

# Initially making a small VGG model just to make sure everything works, before moving to ViT

class VGGModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_blck1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1,
                      padding=0), # ((2 * p + w - k) / s) + 1 --> (maps, 62, 62)
            nn.BatchNorm2d(hidden_units, affine=False),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # ((2 * p + w - k) / s) + 1 --> (maps, 31, 31)s
        )
        self.conv_blck2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0), # (hidden_units, 28, 28)
            nn.BatchNorm2d(hidden_units, affine=False),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0), # (hidden, 26, 26)
            nn.BatchNorm2d(hidden_units, affine=False),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # (hidden, 13, 13)
        )
        self.lin_head = nn.Sequential(
            nn.Flatten(), # (B, C, H, W) -> (B, C*H*W)
            nn.Linear(in_features=hidden_units * 13 * 13, out_features=output_shape)
        )
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.conv_blck1(x)
        x = self.conv_blck2(x)
        x = self.lin_head(x)
        probs = self.softmax(x)
        return x, probs