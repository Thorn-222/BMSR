import os
import sys
import time
import glob
from darts import DataSetDarts
import argparse
import utils
import torch
import torch.nn as nn
import torch.utils
import numpy as np
import logging
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
from utils import convert_to_genotype

from torch.autograd import Variable
from model import NetworkCIFAR as Network

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')#0.05
#parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=500, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=3, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
#parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()
args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        with torch.no_grad():
            input = Variable(input).cuda()
            target = Variable(target).cuda(non_blocking=True)

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def train_each_darts(top_archs):
    CIFAR_CLASSES = 10
    best_acc_list = []
    for i, genotype in enumerate(top_archs):
        genotype = convert_to_genotype(genotype)
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        auxiliary = args.auxiliary
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, auxiliary, genotype)
        model = model.cuda()
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size // 3, shuffle=False, pin_memory=True, num_workers=2)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

        best_acc = 0.0
        for epoch in range(args.epochs):
            scheduler.step()
            logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

            train_acc, train_obj = train(train_queue, model, criterion, optimizer)
            logging.info('train_acc %f', train_acc)

            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            is_best = False
            if valid_acc > best_acc:
                best_acc = valid_acc
                is_best = True
            logging.info('valid_acc %f, best_acc %f', valid_acc, best_acc)
            print('best_acc:',  best_acc)

        best_acc_list.append(best_acc)
        state = {'dataset': top_archs, 'best_acc_list': best_acc_list}
        torch.save(state, os.path.join(args.save, 'darts_test_result.pth.tar'))

    return best_acc_list


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    Darts = DataSetDarts(100)
    best_acc_list = []

    CIFAR_CLASSES = 10

    for genotype in Darts.dataset:
        print(genotype)
        genotype = convert_to_genotype(genotype)
        print(genotype)
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        auxiliary = False
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, auxiliary, genotype)
        model = model.cuda()
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size // 3, shuffle=False, pin_memory=True, num_workers=2)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

        best_acc = 0.0
        for epoch in range(args.epochs):
            scheduler.step()
            logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

            train_acc, train_obj = train(train_queue, model, criterion, optimizer)
            logging.info('train_acc %f', train_acc)

            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            is_best = False
            if valid_acc > best_acc:
                best_acc = valid_acc
                is_best = True
            logging.info('valid_acc %f, best_acc %f', valid_acc, best_acc)

        best_acc_list.append(best_acc)
        state = {'dataset': Darts.dataset, 'best_acc_list': best_acc_list}
        torch.save(state, os.path.join(args.save, 'darts_dataset.pth.tar'))
