import math
import torch
import torch.nn as nn


class PoseNet(nn.Module):
    def __init__(self, original_model):
        super(PoseNet, self).__init__()

        # feature들을 마지막 fully connected layer를 제외화고 ResNet으로 부터 가져옴
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.regressor = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # nn.Linear(2048, 7)
        )

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(0)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, inpt):
        f = self.features(inpt)
        f = f.view(f.size(0), -1)
        y = self.regressor(f)

        return y

