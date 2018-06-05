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

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from tflogger import Logger

from dataset import CIFAR10Dataset

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
parser.add_argument('--use-tflog', default=False, action='store_true',
                    help='log for tensorboard')

best_accu = 0
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
logger = None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class NaiveModel(nn.Module):
    def __init__(self):
        super(NaiveModel, self).__init__()
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
    global args, logger, best_accu, train_losses, train_accuracies, test_losses, test_accuracies
    args = parser.parse_args()

    # create model
    model = NaiveModel().to(device)

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
            best_accu = checkpoint['best_accu']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_losses = checkpoint.get('train_losses', [])
            train_accuracies = checkpoint.get('train_accuracies', [])
            test_losses = checkpoint.get('test_losses', [])
            test_accuracies = checkpoint.get('test_accuracies', [])
            print("=> loaded checkpoint '{}' (epochs {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Decay LR by a factor of 0.1 every 30 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30,
                                          gamma=0.1, last_epoch=args.start_epoch-1)

    # prepare output dir
    # args.output_dir = args.output_dir + '-' + str(time.time())
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    elif not os.path.isdir(args.output_dir):
        print('output dir {} cannot be created'.format(args.output_dir))
        exit(1)

    if args.use_tflog:
        if os.path.exists(os.path.join(args.output_dir, 'logs')):
            shutil.rmtree(os.path.join(args.output_dir, 'logs'))
        logger = Logger(os.path.join(args.output_dir, 'logs'))

    # prepare data
    dataset = CIFAR10Dataset()
    trainloader = dataset.trainloader(args.batch_size, num_workers=2)
    testloader = dataset.testloader(args.batch_size, num_workers=2)
    target_names = dataset.target_names()

    if args.evaluate:
        validate(testloader, model, criterion,
                 verbose=True, target_names=target_names)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_loss, train_accu = train(trainloader, model, criterion,
                                       optimizer, scheduler, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_accu)

        # evaluate on validation set
        test_loss, test_accu = validate(testloader, model, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_accu)

        # remember best prec and save checkpoint
        is_best = test_accu > best_accu
        best_accu = max(test_accu, best_accu)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_accu': best_accu,
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies
        }, is_best)

        print(' >>> Best accuracy: {:.3f} %, at epoch: {}'.format(
            best_accu, test_accuracies.index(best_accu)+1))

    validate(testloader, model, criterion,
             verbose=True, target_names=target_names)
    save_training_plots()


def train(trainloader, model, criterion, optimizer, scheduler, epoch):
    global logger
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    scheduler.step()

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
        y_true = labels.detach().numpy()
        y_pred = np.argmax(outputs.detach().numpy(), axis=1)
        accu = accuracy_score(y_true=y_true, y_pred=y_pred)
        losses.update(loss.item(), inputs.size(0))
        top1.update(accu, inputs.size(0))

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
                  'Accu {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                      epoch, i, len(trainloader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #

            if args.use_tflog:
                # Log scalar values (scalar summary)
                info = {'loss': loss.item(), 'accuracy': accu}

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, i+1)

                # Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), i+1)
                    logger.histo_summary(
                        tag+'/grad', value.grad.data.cpu().numpy(), i+1)

                # Log training images (image summary)
                info = {'images': inputs.view(-1, 3, 32, 32)[:5].cpu().numpy()}

                for tag, images in info.items():
                    logger.image_summary(tag, images, i+1)

    return (losses.avg, top1.avg)


def validate(testloader, model, criterion, verbose=False, target_names=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    y_preds = []
    y_trues = []

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
            y_true = labels.detach().numpy()
            y_pred = np.argmax(outputs.detach().numpy(), axis=1)
            accu = accuracy_score(y_true=y_true, y_pred=y_pred)
            losses.update(loss.item(), inputs.size(0))
            top1.update(accu, inputs.size(0))

            if verbose:
                y_preds.append(y_pred)
                y_trues.append(y_true)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {top1.val:.3f} % ({top1.avg:.3f} %)'
                      .format(i, len(testloader), batch_time=batch_time, loss=losses, top1=top1))

    print(' >>> Accu {top1.avg:.3f} %'.format(top1=top1))

    if verbose:
        y_preds = np.concatenate(y_preds, axis=0)
        y_trues = np.concatenate(y_trues, axis=0)
        print(classification_report(y_true=y_trues,
                                    y_pred=y_preds, target_names=target_names))

    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best):
    global args
    filename = os.path.join(args.output_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.output_dir,
                                               'model_best.pth.tar'))


def save_training_plots():
    global args, train_losses, train_accuracies, test_losses, test_accuracies
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_losses, label='train_loss')
    ax.plot(test_losses, label='test_loss')
    ax.set_xlabel("epoch")
    ax.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss.png'), format='png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_accuracies, label='train_accuracy')
    ax.plot(test_accuracies, label='test_accuracy')
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
