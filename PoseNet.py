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
        self.trans_regressor = nn.Sequential(
            nn.Linear(2048, 3)
        )
        self.rotation_regressor = nn.Sequential(
            nn.Linear(2048, 4)
        )
        self.modelName = 'resnet'

        # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(0)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        for m in self.trans_regressor.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(0)
                m.weight.data[0].normal_(0, 0.5)
                m.weight.data[1].normal_(0, 0.5)
                m.weight.data[2].normal_(0, 0.1)
                m.bias.data.zero_()

        for m in self.rotation_regressor.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(0)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, inpt):
        f = self.features(inpt)
        f = f.view(f.size(0), -1)
        y = self.regressor(f)
        trans = self.trans_regressor(y)
        rotation = self.rotation_regressor(y)

        return trans, rotation

