import time
import os

from util import ramps
import torch.nn as nn
import torch
import logging
from util.utils import *
import math
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from util.losses import FocalLoss
LOG = logging.getLogger('main')
args = None


def set_args(input_args):
    global args
    args = input_args


def train_semi(train_labeled_loader, train_unlabeled_loader, model, ema_model,optimizer, ema_optimizer, all_labels,epoch,criterion, scheduler=None):
    labeled_train_iter = iter(train_labeled_loader)
    unlabeled_train_iter = iter(train_unlabeled_loader)

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()
    end = time.time()
    for i in range(args.epoch_iteration):
        try:
            inputs_x, targets_x, label_index = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(train_labeled_loader)
            inputs_x, targets_x, label_index = labeled_train_iter.next()

        try:
            (inputs_u1, inputs_u2), _, unlabel_index = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(train_unlabeled_loader)
            (inputs_u1, inputs_u2), _, unlabel_index = unlabeled_train_iter.next()

        # measure data loading time
        meters.update('data_time', time.time() - end)
        inputs_x = inputs_x.cuda()
#        inputs_u1 = inputs_u1.cuda()
        inputs_u2 = inputs_u2.cuda()

        batch_size = inputs_x.size(0)
        targets_x_onehot = torch.zeros(batch_size, 33).scatter_(1, targets_x.view(-1, 1), 1)
        targets_x = targets_x.cuda(non_blocking=True)

        targets_u = torch.FloatTensor(all_labels[unlabel_index,:]).cuda()
       
        targets_u = sharpen(targets_u)
        targets_u = targets_u.detach()
        if args.mixup:
            targets_x = targets_x_onehot.cuda(non_blocking=True)
            all_inputs = torch.cat([inputs_x,  inputs_u2], dim=0)
            all_targets = torch.cat([targets_x,  targets_u], dim=0)
            outputs, targets = mixup(all_inputs, all_targets, batch_size, model, epoch + i / args.epoch_iteration)
            loss, class_loss, consistency_loss = semiloss_mixup(outputs, targets,targets_u,targets_u,epoch + i / args.epoch_iteration,criterion)
        else:
            targets_x = targets_x_onehot.cuda(non_blocking=True)
            outputs_x = model(inputs_x)
            outputs_u = model(inputs_u1)
            loss, class_loss, consistency_loss = semiloss(outputs_x, targets_x, outputs_u1, outputs_u2.detach(),epoch + i / args.epoch_iteration)
        meters.update('loss', loss.item())
        meters.update('class_loss', class_loss.item())
        meters.update('cons_loss', consistency_loss.item())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        if scheduler is not None:
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


    ema_optimizer.step(bn=True)
    return meters.averages()['class_loss/avg'], meters.averages()['cons_loss/avg'],all_labels

def train(train_labeled_loader, model, ema_model,optimizer, ema_optimizer,epoch,criterion, scheduler=None):
    labeled_train_iter = iter(train_labeled_loader)


    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()
    end = time.time()
    for i in range(args.epoch_iteration):
        try:
            inputs_x, targets_x, label_index = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(train_labeled_loader)
            inputs_x, targets_x, label_index = labeled_train_iter.next()



        # measure data loading time
        meters.update('data_time', time.time() - end)
        inputs_x = inputs_x.cuda()

        batch_size = inputs_x.size(0)
        targets_x_onehot = torch.zeros(batch_size, 33).scatter_(1, targets_x.view(-1, 1), 1)
        targets_x = targets_x.cuda(non_blocking=True)


        if args.mixup:
            targets_x = targets_x_onehot.cuda(non_blocking=True)
            l = np.random.beta(args.alpha, args.alpha)
            idx = torch.randperm(targets_x.size(0))
            input_a, input_b = inputs_x, inputs_x[idx]
            target_a, target_b = targets_x, targets_x[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            outputs = model(mixed_input)
            
#            loss_mask = torch.max(outputs,dim=1)[0].gt(args.tsa).float().detach()
            loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * mixed_target, dim=1))
#            loss = criterion(outputs,mixed_target)
        else:
            targets_x = targets_x_onehot.cuda(non_blocking=True)
            outputs = model(inputs_x)
            loss, class_loss, consistency_loss = criterion(outputs, outputs)
        meters.update('loss', loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Cons {meters[loss]:.4f}\t'.format(
                    epoch, i, args.epoch_iteration, meters=meters))


    ema_optimizer.step(bn=True)
    return meters.averages()['loss/avg']


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    all_labels = None
    all_outputs = None
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if all_labels is None:
                all_labels = targets
                all_outputs = outputs
            else:
                all_labels = torch.cat([all_labels, targets], dim=0)
                all_outputs = torch.cat([all_outputs, outputs], dim=0)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            if batch_idx % args.print_freq == 0:
                print(
                    '{batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    ))
#    conf_matrix = confusion_matrix(all_outputs, all_labels)
#    plot_confusion_matrix(conf_matrix.numpy(), epoch)
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

def save_checkpoint(name ,state, dirpath, epoch):
    filename = '%s_%d.ckpt' % (name, epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_current_entropy_weight(epoch):
    if epoch > args.consistency_rampup:
        return args.entropy_cost
    else:
        return 0


def sharpen(outputs):
    outputs = outputs ** 2
    outputs = outputs / outputs.sum(dim=1, keepdim=True)
    return outputs


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total,alpha=0.004, cycles=.5, last_epoch=-1):
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
    


def mixup(all_inputs, all_targets, batch_size, model,epoch):
    l = np.random.beta(args.alpha, args.alpha)

    length = get_unsup_size(epoch)
    all_inputs = all_inputs[:args.batch_size+length]
    all_targets = all_targets[:args.batch_size+length]
    idx = torch.randperm(all_inputs.size(0))
    input_a, input_b = all_inputs, all_inputs[idx]
    target_a, target_b = all_targets, all_targets[idx]

    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    outputs = model(mixed_input)
    
    return outputs, mixed_target


def semiloss(outputs_x, targets_x, outputs_u, targets_u,epoch):
    class_loss = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
    consistency_loss = torch.mean(torch.sum(F.softmax(targets_u,1) * (F.log_softmax(targets_u, 1) - F.log_softmax(outputs_u, dim=1)), 1))

    return class_loss + args.consistency_weight*consistency_loss, class_loss, consistency_loss


def semiloss_mixup(outputs_x, targets_x, outputs_u, targets_u, epoch,criterion):
    probs_u = torch.softmax(outputs_u, dim=1)
    class_loss = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
#    class_loss = criterion(outputs_x,targets_x)
    if args.confidence_thresh > 0:
        loss_mask2 = torch.max(probs_u,dim=1)[0].gt(args.confidence_thresh).float().detach() 
        consistency_loss = torch.mean(torch.sum(F.softmax(targets_u,1) * (F.log_softmax(targets_u, 1) - F.log_softmax(outputs_u, dim=1)), 1)*loss_mask2)
    else:
        consistency_loss = torch.mean(torch.sum(F.softmax(targets_u,1) * (F.log_softmax(targets_u, 1) - F.log_softmax(outputs_u, dim=1)), 1))
    if args.entropy_cost >0:
        entropy_cost = ramps.linear_rampup(epoch,args.epochs)*args.entropy_cost
        entropy_loss =- entropy_cost*  torch.mean(torch.sum(torch.mul(F.softmax(outputs_u,dim=1), F.log_softmax(outputs_u,dim=1)),dim=1))
    else:
        entropy_loss = 0
    consistency_weight=args.consistency_weight
#    consistency_loss = torch.mean((torch.softmax(targets_u,dim=1)-probs_u)**2)
#    consistency_weight = ramps.linear_rampup(epoch,args.epochs)*args.consistency_weight
    return class_loss + consistency_weight * consistency_loss + entropy_loss, class_loss, consistency_loss


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def confusion_matrix(preds, labels, n_class=33):
    conf_matrix = torch.zeros(n_class, n_class)
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def get_u_label(model, loader,all_labels):
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, _, index) in enumerate(loader):
            inputs = inputs.cuda()

            # compute output
            outputs = model(inputs)
            all_labels[index] = torch.softmax(outputs,dim=1).cpu().numpy()

    return all_labels

def scheduler(epoch,totals=None,start=0.0,end=1.0):
    if totals is None:
        totals = args.epochs
    step_ratio = epoch/totals
    if args.scheduler == 'linear':
        coeff = step_ratio
    elif args.scheduler == 'exp':
        coeff = np.exp((step_ratio - 1) * 5)
    elif args.scheduler == 'log':
        coeff = 1 - np.exp((-step_ratio) * 5)
    else:
        return 1.0
    return coeff * (end - start) +start

def get_unsup_size(epoch):
    size = int(args.mixup_size*args.batch_size*scheduler(epoch))
    return size
