from train_tool import *
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from util.dataset import get_datasets,RandomErasing,GaussianNoise,ToTensor
from util.ema import ModelEMA
from set_args import create_parser
from util.utils import *
from util.net import ResNet50,TCN
def main(dataset):
    def create_model(ema=False):
        print("=> creating {ema}model ".format(
            ema='EMA ' if ema else ''))

#        model = TCN(input_size=1, output_size=34, num_channels=[32] *8, kernel_size=2)
        model = ResNet50(34)
        model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    global best_prec1

    # Data
    print('==> Preparing tcga data')

    transform_train = transforms.Compose([
        GaussianNoise(), 
        ToTensor(),

    ])
    transform_strong = transforms.Compose([
        RandomErasing(),
        GaussianNoise(),
        ToTensor(),

    ])
    transform_val = transforms.Compose([
        ToTensor(),
    ])

    train_labeled_set, train_unlabeled_set, train_unlabeled_set2, val_set, test_set = get_datasets('./data',args.index, args.n_labeled,  transform_train=transform_train,transform_strong=transform_strong, transform_val=transform_val,withGeo=args.geo)

    train_labeled_loader = data.DataLoader(train_labeled_set, batch_size=args.batch_size,  num_workers=args.num_workers,shuffle=True,drop_last=True)
    train_unlabeled_loader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size*args.unsup_ratio, shuffle=True,
                                            num_workers=args.num_workers, drop_last=True)
    train_unlabeled_loader2 = data.DataLoader(train_unlabeled_set2, batch_size=args.batch_size*args.unsup_ratio, shuffle=False,
                                            num_workers=args.num_workers)

    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = create_model()
    ema_model = ModelEMA(args, model, args.ema_decay)
    criterion = nn.CrossEntropyLoss().cuda()
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=True)

    cudnn.benchmark = True
    totals = args.epochs * args.epoch_iteration
    warmup_step = args.warmup_step * args.epoch_iteration
    scheduler = WarmupCosineSchedule(optimizer, warmup_step, totals)
    all_labels = np.zeros([len(train_unlabeled_set), 34])
    # optionally resume from a checkpoint
    title = dataset
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("Evaluating the  model:")

        test_loss, test_acc = validate(test_loader, model, criterion, args.start_epoch)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        logger = Logger(os.path.join(args.out_path, '%s_log_%d.txt' % (dataset, args.n_labeled)), title=title,
                        resume=True)
        logger.append([args.start_epoch, 0, 0, test_loss, test_acc])
    else:
        logger = Logger(os.path.join(args.out_path, '%s_log_%d.txt' % (dataset, args.n_labeled)), title=title)
        logger.set_names(['epoch', 'Train_class_loss', 'Train_consistency_loss', 'Test_Loss', 'Test_Acc.'])

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        class_loss, cons_loss, all_labels = train_semi(train_labeled_loader, train_unlabeled_loader, model, ema_model,
                                                       optimizer, all_labels, epoch, scheduler)
        all_labels = get_u_label(ema_model.ema, train_unlabeled_loader2, all_labels)

        print("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            print("Evaluating the  model:")
            model.eval()
            test_loss, test_acc = validate(test_loader, model, criterion, epoch)
            print("--- validation in %s seconds ---" % (time.time() - start_time))
            logger.append([epoch, class_loss, cons_loss, test_loss, test_acc])

            print("Evaluating the EMA model:")
            ema_test_loss, ema_test_acc = validate(test_loader, ema_model.ema, criterion, epoch)
            print("--- validation in %s seconds ---" % (time.time() - start_time))
            logger.append([epoch, class_loss, cons_loss, ema_test_loss, ema_test_acc])

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint(
                '%s_%d' % (dataset, args.n_labeled),
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, 'checkpoint_path', epoch + 1)


if __name__ == '__main__':
    dirs = ['result', 'data', 'checkpoint_path']
    for path in dirs:
        if os.path.exists(path) is False:
            os.makedirs(path)
    args = create_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.seed is None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    set_args(args)
    main('TCGA')

