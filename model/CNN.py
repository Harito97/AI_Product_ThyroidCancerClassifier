import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1):
        super(ResidualBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        self.expand_ratio = expand_ratio
        hidden_dim = in_channels * expand_ratio

        # Expansion phase
        self.expand_conv = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=1, stride=1, padding=0
        )
        self.expand_bn = nn.BatchNorm2d(hidden_dim)
        self.expand_relu = nn.ReLU(inplace=True)

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        self.depthwise_swish = nn.SiLU()  # Swish activation

        # Projection phase
        self.project_conv = nn.Conv2d(
            hidden_dim, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.project_bn = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.use_residual = stride == 1 and in_channels == out_channels
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.expand_relu(self.expand_bn(self.expand_conv(x)))
        out = self.depthwise_swish(self.depthwise_bn(self.depthwise_conv(out)))
        out = self.project_bn(self.project_conv(out))

        if self.use_residual:
            out += x
        else:
            out += self.shortcut(x)

        return out


class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        # (input_shape=(-1, 3, 768, 1024)),
        pooling_reducex2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 97, kernel_size=7, stride=2, padding=3),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(97, 97 * 2),
            pooling_reducex2,
            ResidualBlock(97 * 2, 97 * 3),
            pooling_reducex2,
            ResidualBlock(97 * 3, 97 * 4),
            pooling_reducex2,
        )
        self.stage3 = nn.Sequential(
            MBConv(97 * 4, 97 * 5, kernel_size=3, stride=2, expand_ratio=3),
            MBConv(97 * 5, 97 * 6, kernel_size=3, stride=2, expand_ratio=3),
            MBConv(97 * 6, 97 * 7, kernel_size=3, stride=2, expand_ratio=3),
            MBConv(97 * 7, 97 * 8, kernel_size=3, stride=2, expand_ratio=3),
        )
        self.fc1 = nn.Linear(in_features=97 * 8, out_features=97 * 3)
        self.fc2 = nn.Linear(in_features=97 * 3, out_features=num_classes)

    def forward(self, x):
        # print("Before CNN:", x.shape)
        x = self.stage1(x)
        print("After stage1:", x.shape)
        x = self.stage2(x)
        print("After stage2:", x.shape)
        x = self.stage3(x)
        print("After stage3:", x.shape)
        # print("After CNN:", x.shape)
        # print(x)
        x = x.view(x.shape[0], x.shape[1], -1)  # (-1, 97, 12)
        # print("After view:", x.shape)
        x = x.permute(0, 2, 1)
        # print("After permute:", x.shape)
        x = self.fc1(x)
        # print("After fc1:", x.shape)
        x = self.fc2(x)
        # print("After fc2:", x.shape)
        x = F.softmax(x, dim=2)
        # print("After softmax 1:", x.shape)
        # # print(x)
        x = torch.sum(
            x, dim=1, keepdim=False
        )  # Kết quả có kích thước [-1, 3], keepdim=True if want [-1, 1, 3]    # sum or mean all be same (as use the softmax function)
        # # print(x)
        x = F.softmax(x, dim=1)  # Kết quả có kích thước [-1, 3]
        # print("After mean 12 patch and softmax 2:", x.shape)
        return x

    def forward_to_transformer(self, x):
        # print("Before CNN:", x.shape)
        x = self.stage1(x)
        # print("After stage1:", x.shape)
        x = self.stage2(x)
        # print("After stage2:", x.shape)
        x = self.stage3(x)
        # print("After stage3:", x.shape)
        # print("After CNN:", x.shape)
        # # print(x)
        x = x.view(x.shape[0], x.shape[1], -1)  # (-1, 97, 12)
        # print("After view:", x.shape)
        x = x.permute(0, 2, 1)
        # print("After permute:", x.shape)
        x = self.fc1(x)
        # print("After cnn.fc1:", x.shape)
        x = self.fc2(x)
        # print("After cnn.fc2:", x.shape)
        x = F.softmax(x, dim=2)
        # print("After softmax of cnn:", x.shape)
        # print('Done CNN')
        return x


if __name__ == "__main__":
    # Test structure
    model = CNN(num_classes=3)
    # x = torch.randn(2, 3, 768, 1024)
    # y = model(x)
    # print(y.shape)
    # print(y)

    from torchsummary import summary

    summary(model, (3, 768, 1024))

# After stage1: torch.Size([2, 97, 384, 512])
# After stage2: torch.Size([2, 388, 48, 64])
# After stage3: torch.Size([2, 776, 3, 4])
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 97, 384, 512]          14,356
#             Conv2d-2        [-1, 194, 384, 512]          19,012
#        BatchNorm2d-3        [-1, 194, 384, 512]             388
#               ReLU-4        [-1, 194, 384, 512]               0
#             Conv2d-5        [-1, 194, 384, 512]         338,918
#        BatchNorm2d-6        [-1, 194, 384, 512]             388
#               ReLU-7        [-1, 194, 384, 512]               0
#             Conv2d-8        [-1, 194, 384, 512]          37,830
#        BatchNorm2d-9        [-1, 194, 384, 512]             388
#            Conv2d-10        [-1, 194, 384, 512]          19,012
#       BatchNorm2d-11        [-1, 194, 384, 512]             388
#     ResidualBlock-12        [-1, 194, 384, 512]               0
#         MaxPool2d-13        [-1, 194, 192, 256]               0
#            Conv2d-14        [-1, 291, 192, 256]          56,745
#       BatchNorm2d-15        [-1, 291, 192, 256]             582
#              ReLU-16        [-1, 291, 192, 256]               0
#            Conv2d-17        [-1, 291, 192, 256]         762,420
#       BatchNorm2d-18        [-1, 291, 192, 256]             582
#              ReLU-19        [-1, 291, 192, 256]               0
#            Conv2d-20        [-1, 291, 192, 256]          84,972
#       BatchNorm2d-21        [-1, 291, 192, 256]             582
#            Conv2d-22        [-1, 291, 192, 256]          56,745
#       BatchNorm2d-23        [-1, 291, 192, 256]             582
#     ResidualBlock-24        [-1, 291, 192, 256]               0
#         MaxPool2d-25         [-1, 291, 96, 128]               0
#            Conv2d-26         [-1, 388, 96, 128]         113,296
#       BatchNorm2d-27         [-1, 388, 96, 128]             776
#              ReLU-28         [-1, 388, 96, 128]               0
#            Conv2d-29         [-1, 388, 96, 128]       1,355,284
#       BatchNorm2d-30         [-1, 388, 96, 128]             776
#              ReLU-31         [-1, 388, 96, 128]               0
#            Conv2d-32         [-1, 388, 96, 128]         150,932
#       BatchNorm2d-33         [-1, 388, 96, 128]             776
#            Conv2d-34         [-1, 388, 96, 128]         113,296
#       BatchNorm2d-35         [-1, 388, 96, 128]             776
#     ResidualBlock-36         [-1, 388, 96, 128]               0
#         MaxPool2d-37          [-1, 388, 48, 64]               0
#            Conv2d-38         [-1, 1164, 48, 64]         452,796
#       BatchNorm2d-39         [-1, 1164, 48, 64]           2,328
#              ReLU-40         [-1, 1164, 48, 64]               0
#            Conv2d-41         [-1, 1164, 24, 32]          11,640
#       BatchNorm2d-42         [-1, 1164, 24, 32]           2,328
#              SiLU-43         [-1, 1164, 24, 32]               0
#            Conv2d-44          [-1, 485, 24, 32]         565,025
#       BatchNorm2d-45          [-1, 485, 24, 32]             970
#            Conv2d-46          [-1, 485, 24, 32]         188,665
#       BatchNorm2d-47          [-1, 485, 24, 32]             970
#            MBConv-48          [-1, 485, 24, 32]               0
#            Conv2d-49         [-1, 1455, 24, 32]         707,130
#       BatchNorm2d-50         [-1, 1455, 24, 32]           2,910
#              ReLU-51         [-1, 1455, 24, 32]               0
#            Conv2d-52         [-1, 1455, 12, 16]          14,550
#       BatchNorm2d-53         [-1, 1455, 12, 16]           2,910
#              SiLU-54         [-1, 1455, 12, 16]               0
#            Conv2d-55          [-1, 582, 12, 16]         847,392
#       BatchNorm2d-56          [-1, 582, 12, 16]           1,164
#            Conv2d-57          [-1, 582, 12, 16]         282,852
#       BatchNorm2d-58          [-1, 582, 12, 16]           1,164
#            MBConv-59          [-1, 582, 12, 16]               0
#            Conv2d-60         [-1, 1746, 12, 16]       1,017,918
#       BatchNorm2d-61         [-1, 1746, 12, 16]           3,492
#              ReLU-62         [-1, 1746, 12, 16]               0
#            Conv2d-63           [-1, 1746, 6, 8]          17,460
#       BatchNorm2d-64           [-1, 1746, 6, 8]           3,492
#              SiLU-65           [-1, 1746, 6, 8]               0
#            Conv2d-66            [-1, 679, 6, 8]       1,186,213
#       BatchNorm2d-67            [-1, 679, 6, 8]           1,358
#            Conv2d-68            [-1, 679, 6, 8]         395,857
#       BatchNorm2d-69            [-1, 679, 6, 8]           1,358
#            MBConv-70            [-1, 679, 6, 8]               0
#            Conv2d-71           [-1, 2037, 6, 8]       1,385,160
#       BatchNorm2d-72           [-1, 2037, 6, 8]           4,074
#              ReLU-73           [-1, 2037, 6, 8]               0
#            Conv2d-74           [-1, 2037, 3, 4]          20,370
#       BatchNorm2d-75           [-1, 2037, 3, 4]           4,074
#              SiLU-76           [-1, 2037, 3, 4]               0
#            Conv2d-77            [-1, 776, 3, 4]       1,581,488
#       BatchNorm2d-78            [-1, 776, 3, 4]           1,552
#            Conv2d-79            [-1, 776, 3, 4]         527,680
#       BatchNorm2d-80            [-1, 776, 3, 4]           1,552
#            MBConv-81            [-1, 776, 3, 4]               0
#            Linear-82              [-1, 12, 291]         226,107
#            Linear-83                [-1, 12, 3]             876
# ================================================================
# Total params: 12,594,677
# Trainable params: 12,594,677
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 9.00
# Forward/backward pass size (MB): 5222.89
# Params size (MB): 48.04
# Estimated Total Size (MB): 5279.93
# ----------------------------------------------------------------
