import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PoseNet import PoseNet


def main():
    original_model = models.resnet101(pretrained=True)
    model = PoseNet(original_model)
    # model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    cudnn.benchmark = True


if __name__ == '__main__':
    main()
