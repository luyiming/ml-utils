'''
Pytorch version: 0.4.0
'''
import argparse
import os
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='PyTorch Training Template')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-o', '--output-dir', default='model', type=str,
                    help='output directory')

best_prec = 0
train_losses = []
train_precisions = []
test_losses = []
test_precisions = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    global args, best_prec, train_losses, train_precisions, test_losses, test_precisions
    args = parser.parse_args()

    # create model
    model = Net().to(device)

    print('Using', device)

    print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.SGD(model.parameters(), lr=args.lr,
    #                       weight_decay=args.weight_decay,
    #                       momentum=args.momentum,
    #                       nesterov=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if torch.cuda.is_available():
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_losses = checkpoint.get('train_losses', [])
            train_precisions = checkpoint.get('train_precisions', [])
            test_losses = checkpoint.get('test_losses', [])
            test_precisions = checkpoint.get('test_precisions', [])
            print("=> loaded checkpoint '{}' (epochs {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # prepare output dir
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    elif not os.path.isdir(args.output_dir):
        print('output dir {} cannot be created'.format(args.output_dir))
        exit(1)

    # prepare data
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True)
    # [0.49139968  0.48215841  0.44653091]
    train_mean = trainset.train_data.mean(axis=(0, 1, 2)) / 255
    # [0.24703223  0.24348513  0.26158784]
    train_std = trainset.train_data.std(axis=(0, 1, 2)) / 255

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                           download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.evaluate:
        validate(testloader, model, criterion)
        validate_each_class(testloader, model, classes)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_loss, train_prec = train(
            trainloader, model, criterion, optimizer, epoch)
        train_losses.append(train_loss)
        train_precisions.append(train_prec)

        # evaluate on validation set
        test_loss, test_prec = validate(testloader, model, criterion)
        test_losses.append(test_loss)
        test_precisions.append(test_prec)

        # remember best prec and save checkpoint
        is_best = test_prec > best_prec
        best_prec = max(test_prec, best_prec)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_precisions': train_precisions,
            'test_losses': test_losses,
            'test_precisions': test_precisions
        }, is_best)

        print(' >>> Best precision: {:.3f} %, at epoch: {}'.format(
            best_prec, test_precisions.index(best_prec)+1))

    validate_each_class(testloader, model, classes)
    save_training_plots()


def train(trainloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, labels) in enumerate(trainloader):
        # get the inputs
        inputs, labels = inputs.to(device), labels.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        prec = accuracy(outputs, labels)
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                      epoch, i, len(trainloader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

    return (losses.avg, top1.avg)


def validate(testloader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            # get the inputs
            inputs, labels = inputs.to(device), labels.to(device)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            prec = accuracy(outputs, labels)
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec {top1.val:.3f} % ({top1.avg:.3f} %)'.format(
                          i, len(testloader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' >>> Prec {top1.avg:.3f} %'.format(top1=top1))

    return (losses.avg, top1.avg)


def validate_each_class(testloader, model, classes):
    '''
    Parameters:
        classes: int, n_classes
                 list of str, names for each class
    '''
    if isinstance(classes, list) or isinstance(classes, tuple):
        n_classes = len(classes)
    else:
        n_classes = classes
        classes = [i for i in range(n_classes)]

    class_correct = [0.0] * n_classes
    class_total = [0.0] * n_classes
    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(min(args.batch_size, len(labels))):
                class_correct[labels[i]] += c[i].item()
                class_total[labels[i]] += 1
    for i in range(n_classes):
        print('Class {:5} precision: {:.3f} %'.format(
            classes[i], 100 * class_correct[i] / class_total[i]))


def save_checkpoint(state, is_best):
    global args
    filename = os.path.join(args.output_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(
            args.output_dir, 'model_best.pth.tar'))


def accuracy(outputs, labels):
    """Computes the precision"""
    batch_size = labels.size(0)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return 100.0 * correct / batch_size


def save_training_plots():
    global args, train_losses, train_precisions, test_losses, test_precisions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_losses, label='train_loss')
    ax.plot(test_losses, label='test_loss')
    ax.set_xlabel("epoch")
    ax.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss.png'), format='png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_precisions, label='train_accuracy')
    ax.plot(test_precisions, label='test_accuracy')
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy(%)")
    ax.legend()
    plt.savefig(os.path.join(args.output_dir, 'accuracy.png'), format='png')


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
