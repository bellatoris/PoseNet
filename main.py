import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms

import PoseData
from PoseNet import PoseNet


def main():
    # PoseNet의 모델로 resnet101을 사용
    original_model = models.resnet101(pretrained=True)
    # PoseNet 생성
    model = PoseNet(original_model)
    # model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    cudnn.benchmark = True

    # Data loading code
    datadir = './dataset/KingsCollege'

    train_loader = torch.utils.data.DataLoader(
        PoseData.PoseData(datadir, transforms.Compose([
            transforms.Scale(256),
            transforms.RandomCrop(224),
            transforms.ToTensor()
        ]), transforms.ToTensor(), train=True),
        batch_size=75, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        PoseData.PoseData(datadir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]), transforms.ToTensor(), train=False),
        batch_size=75, shuffle=False,
        num_workers=4, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.SGD(model.regressor.parameters(), 1e-5,
                                momentum=0.9,)

    for epoch in range(80):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()
    beta = 700

    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss1 = criterion(output[:, 0:3], target_var[:, 0:3])
        loss2 = criterion(output[:, 3:], target[:, 0:3])
        loss = loss1 + beta * loss2


        print("hi")


def validate(val_loader, model, criterion, optimizer, epoch):
    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss1 = criterion(output[:, 0:3])


    return loss1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 80 epochs"""
    lr = 1e-5 * (0.1 ** (epoch // 80))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
