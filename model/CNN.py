import torch
import torch.nn as nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        # Skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        skip = self.skip_connection(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += skip
        out = self.relu(out)
        out = self.dropout(out)
        return out


class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        # (input_shape=(-1, 3, 768, 1024)),
        self.cnn = nn.Sequential(
            ResidualBlock(in_channels=3, out_channels=7, stride=2),
            ResidualBlock(in_channels=7, out_channels=9, stride=4),
            ResidualBlock(in_channels=9, out_channels=79, stride=4),
            ResidualBlock(in_channels=79, out_channels=97, stride=4),
            ResidualBlock(in_channels=97, out_channels=97, stride=2),
        )
        self.fc1 = nn.Linear(in_features=97, out_features=97)
        self.fc2 = nn.Linear(in_features=97, out_features=num_classes)

    def forward(self, x):
        print("Before CNN:", x.shape)
        x = self.cnn(x)
        print("After CNN:", x.shape)
        # print(x)
        x = x.view(x.shape[0], x.shape[1], -1)  # (-1, 97, 12)
        print("After view:", x.shape)
        x = x.permute(0, 2, 1)
        print("After permute:", x.shape)
        x = self.fc1(x)
        print("After fc1:", x.shape)
        x = self.fc2(x)
        print("After fc2:", x.shape)
        x = F.softmax(x, dim=2)
        print("After softmax 1:", x.shape)
        # print(x)
        x = torch.sum(
            x, dim=1, keepdim=False
        )  # Kết quả có kích thước [-1, 3], keepdim=True if want [-1, 1, 3]    # sum or mean all be same (as use the softmax function)
        # print(x)
        x = F.softmax(x, dim=1)  # Kết quả có kích thước [-1, 3]
        print("After mean 12 patch and softmax 2:", x.shape)
        return x

    def forward_to_transformer(self, x):
        print("Before CNN:", x.shape)
        x = self.cnn(x)
        print("After CNN:", x.shape)
        # print(x)
        x = x.view(x.shape[0], x.shape[1], -1)  # (-1, 97, 12)
        print("After view:", x.shape)
        x = x.permute(0, 2, 1)
        print("After permute:", x.shape)
        x = self.fc1(x)
        print("After cnn.fc1:", x.shape)
        x = self.fc2(x)
        print("After cnn.fc2:", x.shape)
        x = F.softmax(x, dim=2)
        print("After softmax of cnn:", x.shape)
        print('Done CNN')
        return x

if __name__ == "__main__":
    # Test structure
    model = CNN()
    x = torch.randn(3, 3, 768, 1024)
    y = model(x)
    print(y.shape)
    print(y)

    from torchsummary import summary
    summary(model, (3, 768, 1024))

# Before CNN: torch.Size([3, 3, 768, 1024])
# After CNN: torch.Size([3, 97, 3, 4])
# After view: torch.Size([3, 97, 12])
# After permute: torch.Size([3, 12, 97])
# After fc1: torch.Size([3, 12, 97])
# After fc2: torch.Size([3, 12, 3])
# After softmax 1: torch.Size([3, 12, 3])
# After mean 12 patch and softmax 2: torch.Size([3, 3])
# torch.Size([3, 3])
# tensor([[0.4100, 0.4443, 0.1457],
#         [0.4262, 0.4968, 0.0771],
#         [0.2374, 0.5869, 0.1757]], grad_fn=<SoftmaxBackward0>)
# Before CNN: torch.Size([2, 3, 768, 1024])
# After CNN: torch.Size([2, 97, 3, 4])
# After view: torch.Size([2, 97, 12])
# After permute: torch.Size([2, 12, 97])
# After fc1: torch.Size([2, 12, 97])
# After fc2: torch.Size([2, 12, 3])
# After softmax 1: torch.Size([2, 12, 3])
# After mean 12 patch and softmax 2: torch.Size([2, 3])

# Model: "CNN"
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1          [-1, 7, 384, 512]              28
#        BatchNorm2d-2          [-1, 7, 384, 512]              14
#             Conv2d-3          [-1, 7, 384, 512]             196
#        BatchNorm2d-4          [-1, 7, 384, 512]              14
#               ReLU-5          [-1, 7, 384, 512]               0
#            Dropout-6          [-1, 7, 384, 512]               0
#             Conv2d-7          [-1, 7, 384, 512]             448
#        BatchNorm2d-8          [-1, 7, 384, 512]              14
#               ReLU-9          [-1, 7, 384, 512]               0
#           Dropout-10          [-1, 7, 384, 512]               0
#     ResidualBlock-11          [-1, 7, 384, 512]               0
#            Conv2d-12           [-1, 9, 96, 128]              72
#       BatchNorm2d-13           [-1, 9, 96, 128]              18
#            Conv2d-14           [-1, 9, 96, 128]             576
#       BatchNorm2d-15           [-1, 9, 96, 128]              18
#              ReLU-16           [-1, 9, 96, 128]               0
#           Dropout-17           [-1, 9, 96, 128]               0
#            Conv2d-18           [-1, 9, 96, 128]             738
#       BatchNorm2d-19           [-1, 9, 96, 128]              18
#              ReLU-20           [-1, 9, 96, 128]               0
#           Dropout-21           [-1, 9, 96, 128]               0
#     ResidualBlock-22           [-1, 9, 96, 128]               0
#            Conv2d-23           [-1, 79, 24, 32]             790
#       BatchNorm2d-24           [-1, 79, 24, 32]             158
#            Conv2d-25           [-1, 79, 24, 32]           6,478
#       BatchNorm2d-26           [-1, 79, 24, 32]             158
#              ReLU-27           [-1, 79, 24, 32]               0
#           Dropout-28           [-1, 79, 24, 32]               0
#            Conv2d-29           [-1, 79, 24, 32]          56,248
#       BatchNorm2d-30           [-1, 79, 24, 32]             158
#              ReLU-31           [-1, 79, 24, 32]               0
#           Dropout-32           [-1, 79, 24, 32]               0
#     ResidualBlock-33           [-1, 79, 24, 32]               0
#            Conv2d-34             [-1, 97, 6, 8]           7,760
#       BatchNorm2d-35             [-1, 97, 6, 8]             194
#            Conv2d-36             [-1, 97, 6, 8]          69,064
#       BatchNorm2d-37             [-1, 97, 6, 8]             194
#              ReLU-38             [-1, 97, 6, 8]               0
#           Dropout-39             [-1, 97, 6, 8]               0
#            Conv2d-40             [-1, 97, 6, 8]          84,778
#       BatchNorm2d-41             [-1, 97, 6, 8]             194
#              ReLU-42             [-1, 97, 6, 8]               0
#           Dropout-43             [-1, 97, 6, 8]               0
#     ResidualBlock-44             [-1, 97, 6, 8]               0
#            Conv2d-45             [-1, 97, 3, 4]           9,506
#       BatchNorm2d-46             [-1, 97, 3, 4]             194
#            Conv2d-47             [-1, 97, 3, 4]          84,778
#       BatchNorm2d-48             [-1, 97, 3, 4]             194
#              ReLU-49             [-1, 97, 3, 4]               0
#           Dropout-50             [-1, 97, 3, 4]               0
#            Conv2d-51             [-1, 97, 3, 4]          84,778
#       BatchNorm2d-52             [-1, 97, 3, 4]             194
#              ReLU-53             [-1, 97, 3, 4]               0
#           Dropout-54             [-1, 97, 3, 4]               0
#     ResidualBlock-55             [-1, 97, 3, 4]               0
#            Linear-56               [-1, 12, 97]           9,506
#            Linear-57                [-1, 12, 3]             294
# ================================================================
# Total params: 417,772
# Trainable params: 417,772
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 9.00
# Forward/backward pass size (MB): 130.37
# Params size (MB): 1.59
# Estimated Total Size (MB): 140.96
# ----------------------------------------------------------------
