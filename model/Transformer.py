import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, num_classes=3, d_model=3, n_head=3, num_encoder_layers=3):
        # d_model = 3  # Số chiều đặc trưng
        # nhead = 1    # Số đầu của Multihead Attention | d_model % nhead == 0
        # num_encoder_layers = 3  # Số lớp Encoder
        # num_classes = 3  # Số lớp phân loại
        super(Transformer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dropout=0.3,
            activation=F.relu,
            device=self.device,
            dim_feedforward=97,
        )
        # Only use encoder for the classification task
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )  # enable_nested_tensor=False
        # as the n_head is odd then can't use nested_tensor efficiently
        # (the warning will be shown)
        self.nb = nn.BatchNorm1d(num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(in_features=d_model, out_features=num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model] -> [seq_len, batch_size, d_model]
        x = x.permute(1, 0, 2)
        # print("Before Transformer Encoder:", x.shape)
        x = self.transformer_encoder(x)
        # print("After Transformer Encoder:", x.shape)
        # Use the output of the [0] index (representing the first token) for classification
        x = x.mean(dim=0)  # Aggregate the output over sequence length
        x = self.nb(x)
        x = self.dropout(x)
        # print("After mean:", x.shape)
        x = self.fc(x)
        # print("After fc:", x.shape)
        x = F.softmax(x, dim=-1)  # Softmax over the last dimension (or also x[1])
        # print("After softmax:", x.shape)
        # print("Done Transformer")
        return x


if __name__ == "__main__":
    # Define the model
    model = Transformer()
    x = torch.randn(3, 12, 3)  # (batch_size, seq_len, d_model)
    y = model(x)
    # print(y.shape)
    # print(y)


# from torchinfo import (
#     summary,
# )  # Thay vì torchsummary (can use for transformer model)

# summary(model, input_size=(3, 12, 3))  # (batch_size, seq_len, d_model)
# # summary should be run outside the main function to show the model structure
# ===============================================================================================
# Layer (type:depth-idx)                        Output Shape              Param #
# ===============================================================================================
# Transformer                                   [3, 3]                    --
# ├─TransformerEncoder: 1-1                     [12, 3, 3]                --
# │    └─ModuleList: 2-1                        --                        --
# │    │    └─TransformerEncoderLayer: 3-1      [12, 3, 3]                742
# │    │    └─TransformerEncoderLayer: 3-2      [12, 3, 3]                742
# │    │    └─TransformerEncoderLayer: 3-3      [12, 3, 3]                742
# ├─BatchNorm1d: 1-2                            [3, 3]                    6
# ├─Dropout: 1-3                                [3, 3]                    --
# ├─Linear: 1-4                                 [3, 3]                    12
# ===============================================================================================
# Total params: 2,244
# Trainable params: 2,244
# Non-trainable params: 0
# Total mult-adds (Units.MEGABYTES): 0.03
# ===============================================================================================
# Input size (MB): 0.00
# Forward/backward pass size (MB): 0.09
# Params size (MB): 0.01
# Estimated Total Size (MB): 0.10
# ===============================================================================================
