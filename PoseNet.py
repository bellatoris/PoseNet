import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import ResNet


class PoseNet(nn.Module):
    def __init__(self, original_model):
        super(PoseNet, self).__init__()

        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.regressor = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.Linear(2048, 7)
        )
        self.modelName = 'resnet'

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

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

