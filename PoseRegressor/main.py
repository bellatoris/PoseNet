import shutil
import numpy as np
import math

import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms

from PoseRegressor.SeqPoseData import SeqPoseData
from PoseRegressor.RegNet import RegNet


def main():
    best_loss = 10000
    start_epoch = 0
    train_batch_size = 32

    # PoseNet의 모델로 resne34을 사용
    original_model = models.resnet34(pretrained=True)
    # PoseNet 생성
    model = RegNet(original_model, batch_size=train_batch_size, seq_length=5, gru_layer=1)
    # model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # for resume code
    # checkpoint = torch.load('reg_checkpoint.pth.tar-Res34')
    # model.load_state_dict(checkpoint['state_dict'])
    # start_epoch = checkpoint['epoch']
    # best_loss = checkpoint['best_loss']

    cudnn.benchmark = True

    # Data loading code
    datadir = '../dataset'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # train data loader, random cropping and scaling, shuffling
    train_loader = torch.utils.data.DataLoader(
        SeqPoseData(datadir, seq_length=5, transform=transforms.Compose([
            transforms.Scale(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize
        ]), train=True),
        batch_size=train_batch_size, shuffle=True,
        num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        SeqPoseData(datadir, seq_length=5, transform=transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]), train=False),
        batch_size=train_batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    lr = 1e-4
    optimizer = torch.optim.Adam([{'params': model.rnn.parameters(), 'lr': lr},
                                  {'params': model.features.parameters(), 'lr': lr},
                                  {'params': model.trans_regressor.parameters(), 'lr': lr},
                                  {'params': model.rotation_regressor.parameters(), 'lr': lr}],
                                 weight_decay=2e-4)

    for epoch in range(start_epoch, 160):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, train_batch_size)

        # evaluate on validation set
        loss, trans_loss, rotation_loss = validate(val_loader, model, train_batch_size)

        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
        }, is_best)


def train(train_loader, model, optimizer, epoch, batch_size):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    trans_losses = AverageMeter()
    rotation_losses = AverageMeter()

    # switch to train mode
    model.train()
    beta = 100

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # because of gru we need to fix the batch_size
        if input.size(0) != batch_size:
            continue
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())

        # compute output
        trans_output, rotation_output = model(input_var)
        trans_loss = pose_loss(trans_output, target_var[:, 4, :3]) * 10
        rotation_loss = pose_loss(rotation_output, target_var[:, 4, 3:]) * beta
        loss = trans_loss + rotation_loss

        # measure and record loss
        losses.update(loss.data[0], input.size(0))
        trans_losses.update(trans_loss.data[0], input.size(0))
        rotation_losses.update(rotation_loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Trans Loss {trans_loss.val:.4f} ({trans_loss.avg:.4f})\t'
              'Rotation Loss {rotation_loss.val:.4f} ({rotation_loss.avg:.4f})\t'.format(
               epoch, i, len(train_loader), batch_time=batch_time,
               loss=losses, data_time=data_time, trans_loss=trans_losses,
               rotation_loss=rotation_losses))


def validate(val_loader, model, batch_size):
    losses = AverageMeter()
    trans_losses = AverageMeter()
    rotation_losses = AverageMeter()
    rotation_errors = AverageMeter()

    # switch to evaluate mode
    model.eval()
    beta = 10

    for i, (input, target) in enumerate(val_loader):
        if input.size(0) != batch_size:
            continue
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target.cuda(), volatile=True)

        # compute output
        trans_output, rotation_output = model(input_var)
        trans_loss = pose_loss(trans_output, target_var[:, 4, 0:3])
        rotation_loss = pose_loss(rotation_output, target_var[:, 4, 3:]) * beta
        loss = trans_loss + rotation_loss

        # measure and record loss
        losses.update(loss.data[0], input.size(0))
        trans_losses.update(trans_loss.data[0], input.size(0))
        rotation_losses.update(rotation_loss.data[0], input.size(0))
        rotation_errors.update(rotation_error(rotation_output, target_var[:, 4, 3:]).data[0],
                               input.size(0))

    print('Test: [{0}]\t'
          'Loss ({loss.avg:.4f})\t'
          'Trans Loss ({trans_loss.avg:.4f})\t'
          'Rotation Loss ({rotation_loss.avg:.4f})\t'
          'Rotation Error ({rotation_error.avg:.4f})\t'.format(
           len(val_loader), loss=losses,
           trans_loss=trans_losses, rotation_loss=rotation_losses,
           rotation_error=rotation_errors))

    return losses.avg, trans_losses.avg, rotation_losses.avg


def save_checkpoint(state, is_best, filename='reg_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'reg_model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 80 epochs"""
    lr = 1e-4 * (0.1 ** (epoch // 80))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pose_loss(input, target):
    """Gets l2 loss between input and target"""
    x = torch.norm(input-target, dim=1)
    x = torch.mean(x)

    return x


def rotation_error(input, target):
    """Gets cosine distance between input and target """
    x1 = torch.norm(input, dim=1)
    x2 = torch.norm(target, dim=1)

    x1 = torch.div(input, torch.stack((x1, x1, x1, x1), dim=1))
    x2 = torch.div(target, torch.stack((x2, x2, x2, x2), dim=1))
    d = torch.abs(torch.sum(x1 * x2, dim=1))
    theta = 2 * torch.acos(d) * 180/math.pi
    theta = torch.mean(theta)

    return theta


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
