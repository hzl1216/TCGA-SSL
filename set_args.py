import argparse
from utils import str2bool
def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch tcga Training')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        metavar='N', help='labeled-batch size')
    parser.add_argument('--unsup-ratio', default=7, type=int,
                        metavar='N', help='The ratio between batch size of unlabeled data and labeled data')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float)
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency-weight', default=0.0, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE',
                        choices=['mse', 'kl'],
                        help='consistency loss type to use')
    parser.add_argument('--consistency-rampup', default=5, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--entropy-cost', default=0.0, type=float, metavar='WEIGHT')
    parser.add_argument('--checkpoint-epochs', default=1, type=int,
                        metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('--evaluation-epochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--print-freq', '-p', default=128, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--out_path', default='result',
                        help='Directory to output the result')
    parser.add_argument('--n-labeled', type=int, default=4000,
                        help='Number of labeled data')
    parser.add_argument('-e', '--evaluate', type=bool,
                        help='evaluate model on evaluation set')
    parser.add_argument('--num-workers', type=int, default=12,
                        help='Number of workers')
    parser.add_argument('--epoch-iteration', type=int, default=1024,
                        help='train step of one epoch')
    parser.add_argument('--warmup-step', type=int, default=10,
                        help='Number of workers')
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--mixup', default=True, type=str2bool,
                        help='use mixup', metavar='BOOL')
    parser.add_argument('--gpu', default='1', type=str,
                help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', type=int, default=42, help='manual seed')
    parser.add_argument('--confidence-thresh', default=-1,type=float)
    parser.add_argument('--scheduler', default='linear')
    parser.add_argument('--ema-stage', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--val-size', type=int, default=-1)
    parser.add_argument('--mixup-size', type=int, default=7)
    parser.add_argument('--tsa', type=int, default=0.9)
    parser.add_argument('--dataset', type=str, default='tcga')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--geo', default=False, type=str2bool,
                        help='use geo dataset as unlabeled set', metavar='BOOL')
    return parser.parse_args()
