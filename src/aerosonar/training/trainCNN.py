import os
import sys
import platform
import ctypes
# Force Windows to search the torch/lib directory for DLL dependencies
if platform.system() == "Windows":
    # Replace this with the ACTUAL path to your .venv torch lib folder
    dll_path = r"C:\Programming\Python\VScode\Aerial Detector\.venv\Lib\site-packages\torch\lib\c10.dll"
    
    try:
        if os.path.exists(dll_path):
            # Pre-load the DLL into memory before torch tries to
            ctypes.CDLL(os.path.normpath(dll_path))
            print("Successfully pre-loaded c10.dll")
    except Exception as e:
        print(f"Pre-load failed: {e}")

import torch
print(f"Is CUDA available: {torch.cuda.is_available()}")



import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from aerosonar.models.spectrogramCNN import SpectrogramCNN
from aerosonar.data.dataset import *
LR = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SpectrogramCNN(freq_bins=128, time_frames=87, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), LR, [0.9,0.99], 1e-10)


import time

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    losses_list = []
    errors = []
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        errors.append(100 - acc1[0].item())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        losses_list.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.display(i)

    return top1.avg, top5.avg, losses_list, errors

def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    losses_list = []
    errors = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            errors.append(100 - acc1[0].item())
            losses_list.append(loss.item())
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses_list, errors


EPOCHS = 10

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=1)


train_losses = []
train_acc1 = []
train_acc5 = []
test_losses = []
test_acc1 = []
test_acc5 = []
test_error = []
train_error = []

for epoch in range(0, EPOCHS):

    # train for one epoch
    acc1, acc5, losses, error = train(train_loader, net, criterion, optimizer, epoch)
    train_acc1.append(acc1.item())
    train_acc5.append(acc5.item())
    train_losses.append(sum(losses)/len(losses))
    train_error.extend(error)

    # evaluate on validation set
    acc1, acc5, losses, error = validate(test_loader, net, criterion)
    test_acc1.append(acc1.item())
    test_acc5.append(acc5.item())
    test_losses.append(sum(losses)/len(losses))
    test_error.extend(error)

    # scheduler.step()
    avg_loss = sum(losses) / len(losses)
    scheduler.step(avg_loss)