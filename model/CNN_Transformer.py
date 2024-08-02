import torch
import torch.nn as nn
import torch.nn.functional as F

from CNN import CNN
from Transformer import Transformer

# Định nghĩa mô hình
class CNN_Transformer(nn.Module):
    def __init__(self, d_model=3, nhead=3, num_encoder_layers=3, num_classes=3):
        super(CNN_Transformer, self).__init__()
        # (input_shape=(3, 768, 1024)),
        self.cnn = CNN(num_classes=num_classes)

        self.fc1 = nn.Linear(in_features=97, out_features=3)

        self.transformer_encoder = Transformer(
            num_classes=num_classes,
            d_model=d_model,
            n_head=nhead,
            num_encoder_layers=num_encoder_layers,
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.cnn.forward_to_transformer(x)

        x = self.transformer_encoder(x)
        print("After Transformer Encoder:", x.shape)
        return x

if __name__ == "__main__":
    # Define the model
    model = CNN_Transformer()
    x = torch.randn(3, 3, 768, 1024)  # (batch_size, seq_len, d_model)
    y = model(x)
    print(y.shape)
    print(y)

# Before CNN: torch.Size([3, 3, 768, 1024])
# After CNN: torch.Size([3, 97, 3, 4])
# After view: torch.Size([3, 97, 12])
# After permute: torch.Size([3, 12, 97])
# After cnn.fc1: torch.Size([3, 12, 97])
# After cnn.fc2: torch.Size([3, 12, 3])
# After softmax of cnn: torch.Size([3, 12, 3])
# Done CNN
# Before Transformer Encoder: torch.Size([12, 3, 3])
# After Transformer Encoder: torch.Size([12, 3, 3])
# After mean: torch.Size([3, 3])
# After fc: torch.Size([3, 3])
# After softmax: torch.Size([3, 3])
# Done Transformer
# After Transformer Encoder: torch.Size([3, 3])
# torch.Size([3, 3])
# tensor([[0.8029, 0.0149, 0.1822],
#         [0.1599, 0.1726, 0.6674],
#         [0.0799, 0.4685, 0.4516]], grad_fn=<SoftmaxBackward0>)
