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

    # Data loading code
    traindir = os.path.join('./dataset', 'train')

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor()
        ])),
        batch_szize=75, num_workers=4, pin_memory=True
    )

if __name__ == '__main__':

    main()
