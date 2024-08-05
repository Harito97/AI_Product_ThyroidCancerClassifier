import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader, Dataset

class H97_1__0_1_2(nn.Module):
    def __init__(self, num_classes=3):
        super(H97_1__0_1_2, self).__init__()
        # Khởi tạo mô hình với trọng số tiền huấn luyện
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT) # pretrained=True)
        # Thay thế lớp phân loại cuối cùng
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # Xây dựng model
    model = H97_1__0_1_2(num_classes=3)
    print(model)
