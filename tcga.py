from train_tool import *
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from util.dataset import get_datasets,RandomErasing,GaussianNoise,ToTensor
from set_args import create_parser
from util.net import ResNet50,TCN
def main():
    def create_model(ema=False):
        print("=> creating {ema}model ".format(
            ema='EMA ' if ema else ''))

#        model = TCN(input_size=1, output_size=33, num_channels=[32] *8, kernel_size=2)
        model = ResNet50(33)
        model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    global best_prec1

    # Data
    print('==> Preparing tcga data')

    transform_train = transforms.Compose([
#        RandomErasing(),
        GaussianNoise(), 
        ToTensor(),

    ])
    transform_strong = transforms.Compose([
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
    if args.val_size > 0:
        val_loader = data.DataLoader(val_set, batch_size=args.batch_size*args.unsup_ratio, shuffle=False, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = create_model()
    ema_model = create_model(ema=True)
    tmp_model= create_model(ema=True)

    criterion = nn.CrossEntropyLoss().cuda()
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    ema_optimizer = WeightEMA(model, ema_model, tmp_model, alpha=args.ema_decay)
    cudnn.benchmark = True
    if args.warmup_step>0:
        totals = args.epochs*args.epoch_iteration
        warmup_step = args.warmup_step*args.epoch_iteration
        scheduler =  WarmupCosineSchedule(optimizer,warmup_step,totals)
    else:
        scheduler = None
    all_labels = np.zeros([len(train_unlabeled_set), 33])
    # optionally resume from a checkpoint
    title = 'tcga'
    best_acc = 0
    best_epoch = 0
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Evaluating the  model:")
        if args.val_size > 0:
            val_loss, val_acc = validate(val_loader, model, criterion)
        else:
            val_loss, val_acc = 0, 0
        test_loss, test_acc = validate(test_loader, model, criterion)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        logger = Logger(os.path.join(args.out_path, '%s_log_%d_%d.txt'%(title,args.n_labeled,args.index)), title=title, resume=True)
        logger.append([args.start_epoch, 0, 0, val_loss, val_acc,test_loss, test_acc])
    else:
        logger = Logger(os.path.join(args.out_path, '%s_log_%d_%d.txt'%(title,args.n_labeled,args.index)), title=title)
        logger.set_names(['epoch', 'Train_class_loss',  'Train_consistency_loss', 'Val_Loss', 'Val_Acc.', 'Test_Loss', 'Test_Acc'])

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        
        if epoch >= args.ema_stage:
            print('train in semi-supervised stage2')
            all_labels = get_u_label(ema_model, train_unlabeled_loader2, all_labels)
            class_loss, cons_loss = train_semi(train_labeled_loader, train_unlabeled_loader, model, ema_model,optimizer,ema_optimizer, all_labels, epoch, criterion,scheduler)
        else:
            print(' train in semi-supervised stage1')
            cons_loss = 0
            class_loss = train(train_labeled_loader, model, ema_model,optimizer, ema_optimizer, epoch, criterion, scheduler)
        print("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            print("Evaluating the  model:")
            if args.val_size>0:
                val_loss, val_acc = validate(val_loader, model, criterion)
            else:
                val_loss, val_acc = 0, 0
            test_loss, test_acc = validate(test_loader, model, criterion)
            print("--- validation in %s seconds ---" % (time.time() - start_time))
            logger.append([epoch, class_loss, cons_loss, val_loss, val_acc,test_loss, test_acc])
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch=epoch
            if epoch > best_epoch+100:
                break
            print("Evaluating the EMA model:")
            if args.val_size > 0:
                ema_val_loss, ema_val_acc = validate(val_loader, ema_model, criterion)
            else:
                ema_val_loss, ema_val_acc = 0, 0
            ema_test_loss, ema_test_acc = validate(test_loader, ema_model, criterion)
            print("--- validation in %s seconds ---" % (time.time() - start_time))
            logger.append([epoch, class_loss, cons_loss, ema_val_loss, ema_val_acc,ema_test_loss, ema_test_acc])

    save_checkpoint(
                '%s_%d'%(title, args.n_labeled),
                {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'checkpoint_path', args.index)

if __name__ == '__main__':
    dirs = ['result', 'data', 'checkpoint_path']
    for path in dirs:
        if os.path.exists(path) is False:
            os.makedirs(path) 
    args = create_parser()
    print('train in %d fold data'%args.index)
#    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 
    torch.cuda.manual_seed(args.seed)
    set_args(args)
    main()
