import math
import torch.nn as nn


class PoseNet(nn.Module):
    def __init__(self, original_model):
        super(PoseNet, self).__init__()

        # feature들을 마지막 fully connected layer를 제외화고 ResNet으로 부터 가져옴
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.regressor = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.Linear(2048, 7)
        )
        self.modelName = 'resnet'

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

        # Linear layer들도 He initialization을 사용한다
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(0)
                m.weight.data.normal_(0, math.sqrt(1./n))
                m.bias.data.zero_()

    def forward(self, inpt):
        f = self.features(inpt)
        f = f.view(f.size(0), -1)
        y = self.regressor(f)

        return y

