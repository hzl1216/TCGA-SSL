import time
import os

import torch.nn as nn
import torch
import logging
from util.utils import *
import math
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

LOG = logging.getLogger('main')
args = None


def set_args(input_args):
    global args
    args = input_args


def train_semi(train_labeled_loader, train_unlabeled_loader, model, ema_model, optimizer, all_labels, epoch,
               scheduler=None):
    labeled_train_iter = iter(train_labeled_loader)
    unlabeled_train_iter = iter(train_unlabeled_loader)

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    end = time.time()
    for i in range(args.epoch_iteration):
        try:
            inputs_x, targets_x, label_index = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(train_labeled_loader)
            inputs_x, targets_x, label_index = labeled_train_iter.next()

        try:
            (inputs_aug, inputs_std), _, unlabel_index = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(train_unlabeled_loader)
            (inputs_aug, inputs_std), _, unlabel_index = unlabeled_train_iter.next()

        # measure data loading time
        meters.update('data_time', time.time() - end)
        inputs_x = inputs_x.cuda()
        inputs_aug = inputs_aug.cuda()
        inputs_std = inputs_std.cuda()

        batch_size = inputs_x.size(0)
        targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1, 1), 1).cuda(non_blocking=True)

        targets_u = torch.FloatTensor(all_labels[unlabel_index, :]).cuda()
        targets_u = sharpen(targets_u)
        targets_u = targets_u.detach()

        if args.mixup:
            mixup_size = get_mixup_size(epoch + i / args.epoch_iteration)

            inputs_x = torch.cat([inputs_x, inputs_std[:mixup_size]], dim=0)
            targets_x = torch.cat([targets_x, targets_u[:mixup_size]], dim=0)
            mixup_size += args.batch_size
            l = np.random.beta(args.alpha, args.alpha)
            idx = torch.randperm(inputs_x.size(0))
            input_b = inputs_x[idx]
            target_b = targets_x[idx]
            mixed_inputs = l * inputs_x + (1 - l) * input_b
            mixed_targets = l * targets_x + (1 - l) * target_b
            del inputs_x, targets_x, input_b, target_b
            all_inputs = torch.cat([mixed_inputs, inputs_aug, inputs_std])

            all_logits = model(all_inputs)
            logits_aug, logits_std = all_logits[mixup_size:].chunk(2)
            logits_mixup = all_logits[:mixup_size]
            del all_logits
            loss, class_loss, consistency_loss = semiloss_mixup(logits_mixup, mixed_targets, logits_aug,
                                                                logits_std.detach())

        meters.update('loss', loss.item())
        meters.update('class_loss', class_loss.item())
        meters.update('cons_loss', consistency_loss.item())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_model.update(model)
        scheduler.step()
        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Cons {meters[cons_loss]:.4f}\t'.format(
                    epoch, i, args.epoch_iteration, meters=meters))
    return meters.averages()['class_loss/avg'], meters.averages()['cons_loss/avg'], all_labels


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            logits = model(inputs)
            loss = criterion(logits, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits, targets, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress

        print(
            'Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            ))

    return losses.avg, top1.avg


class WeightEMA(object):
    def __init__(self, model, ema_model, tmp_model=None, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        if tmp_model is not None:
            self.tmp_model = tmp_model.cuda()
        self.wd = args.weight_decay

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                if args.optimizer == 'Adam':
                    param.data.mul_(1 - self.wd)


def save_checkpoint(name, state, dirpath, epoch):
    filename = '%s_%d.ckpt' % (name, epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)


def sharpen(logits):
    logits = logits ** 2
    logits = logits / logits.sum(dim=1, keepdim=True)
    return logits


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, alpha=0.004, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.alpha = alpha
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        cosine_decay = 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return decayed


def semiloss(logits_x, targets_x, logits_u, targets_u):
    class_loss = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * targets_x, dim=1))
    consistency_loss = torch.mean(
        torch.sum(F.softmax(targets_u, 1) * (F.log_softmax(targets_u, 1) - F.log_softmax(logits_u, dim=1)), 1))

    return class_loss + args.consistency_weight * consistency_loss, class_loss, consistency_loss


def semiloss_mixup(logits_x, targets_x, logits_u, targets_u):
    class_loss = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * targets_x, dim=1))
    consistency_loss = torch.mean(
        torch.sum(F.softmax(targets_u, 1) * (F.log_softmax(targets_u, 1) - F.log_softmax(logits_u, dim=1)), 1))

    return class_loss + args.consistency_weight * consistency_loss, class_loss, consistency_loss


def get_u_label(model, loader, all_labels):
    with torch.no_grad():
        for batch_idx, (inputs, _, index) in enumerate(loader):
            inputs = inputs.cuda()

            # compute output
            logits = model(inputs)
            all_labels[index] = torch.softmax(logits, dim=1).cpu().numpy()

    return all_labels


def scheduler(epoch, start=0.0, end=1.0):
    totals = args.epochs
    step_ratio = epoch / totals
    if args.scheduler == 'linear':
        coeff = step_ratio
    elif args.scheduler == 'exp':
        coeff = np.exp((step_ratio - 1) * 5)
    elif args.scheduler == 'log':
        coeff = 1 - np.exp((-step_ratio) * 5)
    else:
        return 1.0
    return coeff * (end - start) + start


def get_mixup_size(epoch):
    size = int(args.mixup_size * args.batch_size * scheduler(epoch) / 4) * 4
    return size
