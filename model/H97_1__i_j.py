import torch
import torch.nn as nn

class H97_1__i_j(nn.Module):
    def __init__(self, num_classes=3):
        super(H97_1__i_j, self).__init__()
        self.fc2 = nn.Linear(num_classes, 2)

    def forward(self, x):
        # x = self.model(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x