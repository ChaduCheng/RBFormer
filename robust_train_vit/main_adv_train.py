
import argparse
import datetime
import json
import sys
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import models
import utils
from utils import get_logger
from pathlib import Path
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma
from torch.optim import SGD, lr_scheduler

from datasets import build_dataset
from engine import train_one_epoch, evaluate, robust_evaluate
from losses import DistillationLoss
from model_adv_att import AttackPGD
from samplers import RASampler
from vit_pytorch.Robust_VITmodel import Robust_ViTmodel
from vit_pytorch.imageAgg import ImageAgg
from vit_pytorch.cnn_mod import CNNVIT
from vit_pytorch.swin import SwinTransformer



def get_args_parser():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=90, type=int)

    # Model parameters
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')  ## change
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='./imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'TinyIMN', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./test/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # adversarial training parameters
    parser.add_argument('--advtrain', default=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    ### model elements


    parser.add_argument('--patch', default='4', type=int)

    ### different choice here：  48
    ## 1. embedding: 1） img_input; 2) conv_input   2
    ## 2. Trans_block: 1) ori_att 2) MLP_att    2
    ## 3. MLP 1) ori_MLP 2） conv_MLP 3) no_MLP   3
    ## 4. norm: 1) LN 2) none         2
    ## 5. skipconnect: 1) residual 2） none    2

    parser.add_argument('--embedding', default='conv_input', help='which embedding using here')
    parser.add_argument('--Transblock', default='ori_att', help='which embedding using here')
    parser.add_argument('--MLP', default='ori_MLP', help='which embedding using here')
    parser.add_argument('--norm', default='LN', help='which embedding using here')
    parser.add_argument('--skipconnect', default='residual', help='which embedding using here')








    return parser


def main(args):
    utils.init_distributed_mode(args)
    # print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    print('clla the output_class:', args.nb_classes)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        # batch_size=int(1.5 * args.batch_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # print(f"Creating model: {args.model}")

    parts = {'embedding': args.embedding, 'Trans_block': args.Transblock, 'Trans_MLP': args.MLP, 'norm': args.norm,
             'skipconnect': args.skipconnect}

    # model = Robust_ViTmodel(
    #     image_size=32,
    #     patch_size=args.patch,
    #     num_classes=10,
    #     dim=512,
    #     depth=12,
    #     heads=16,
    #     mlp_dim=512,
    #     dropout=0.1,
    #     emb_dropout=0.1,
    #     parts=parts
    # ).cuda()

    model = ImageAgg(
        image_size=224,
        patch_size=8,
        dim=96,
        heads=3,  ### 8
        num_hierarchies=3,  # number of hierarchies
        block_repeats=(2, 2, 8),  # the number of transformer blocks at each heirarchy, starting from the bottom
        num_classes=1000
    )
   ## CNN based
#     model = CNNVIT(
#     num_classes = 10,
#     s1_emb_dim = 64,        # stage 1 - dimension
#     s1_emb_kernel = 7,      # stage 1 - conv kernel
#     s1_emb_stride = 4,      # stage 1 - conv stride
#     s1_proj_kernel = 3,     # stage 1 - attention ds-conv kernel size
#     s1_kv_proj_stride = 2,  # stage 1 - attention key / value projection stride
#     s1_heads = 1,           # stage 1 - heads
#     s1_depth = 1,           # stage 1 - depth
#     s1_mlp_mult = 4,        # stage 1 - feedforward expansion factor
#     s2_emb_dim = 192,       # stage 2 - (same as above)
#     s2_emb_kernel = 3,
#     s2_emb_stride = 2,
#     s2_proj_kernel = 3,
#     s2_kv_proj_stride = 2,
#     s2_heads = 3,
#     s2_depth = 2,
#     s2_mlp_mult = 4,
#     s3_emb_dim = 384,       # stage 3 - (same as above)
#     s3_emb_kernel = 3,
#     s3_emb_stride = 2,
#     s3_proj_kernel = 3,
#     s3_kv_proj_stride = 2,
#     s3_heads = 4,
#     s3_depth = 10,
#     s3_mlp_mult = 4,
#     dropout = 0.
# ).cuda()

    ## swin based
    # model = swin_t().cuda()

    output_dir_all = args.output_dir

    output_dir = Path(output_dir_all)

    if output_dir_all:
        output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir_all + 'epx.log'

    logger = get_logger(log_file)


    logger.info('model structure')



    # attention = model.transformer.layers[1]
    # embedding = model.to_patch_embedding
    #
    # print('ViT elements', parts)
    #
    # logger.info('ViT elements: {}'.format(parts))
    #
    # print('embedding:', embedding)
    #
    # logger.info('embedding: {}'.format(embedding))
    #
    # print('attention is:', attention)
    #
    # logger.info('attention is:{}'.format(attention))

    print('the model is:', model)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    # lr_scheduler, _ = create_scheduler(args, optimizer)

    # optimizer = SGD(model.parameters(), 0.1, momentum=0.9,
    #                 weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )



    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            print('resume from 1 here:', args.resume)
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print('resume from 2 here')
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, logger)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    # Initialize attacker
    # cifar10_mean = (0.4914, 0.4822, 0.4465)
    # cifar10_std = (0.2471, 0.2435, 0.2616)
    #
    # mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
    # std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    mu = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1).cuda()
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1).cuda()



    config_linf = {
        'epsilon': 4.0 / 255 / std,
        'num_steps': 3,
        # 'step_size': 2.0 / 255,
        # 'step_size': 0.01,
        'step_size': 4.0 * 2 / 255 / std / 3,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'linf'
    }

    attacker = AttackPGD(model, config=config_linf).to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_robust_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        ## train here
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning,
            adv_train=args.advtrain,
            attacker=attacker
        )

        # lr_scheduler.step(epoch)
        scheduler.step(epoch)


        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'epoch': epoch,
                    # 'optimizer': optimizer.state_dict(),
                    # 'lr_scheduler': scheduler,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        # if args.output_dir:
        #     checkpoint_paths = [output_dir / 'checkpoint.pth']
        #     for checkpoint_path in checkpoint_paths:
        #         utils.save_on_master({
        #             'model': model_without_ddp.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'lr_scheduler': lr_scheduler.state_dict(),
        #             'epoch': epoch,
        #             'model_ema': get_state_dict(model_ema),
        #             'scaler': loss_scaler.state_dict(),
        #             'args': args,
        #         }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device,logger)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")



        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        print(f'Max clean accuracy: {max_accuracy:.2f}%')
        logger.info(f'Max clean accuracy: {max_accuracy:.2f}%')
        # logger.info('Max clean accuracy: {max_accuracy:.2f}%'.format(max_accuracy=max_accuracy))



        if args.advtrain:

            robust_test_stats = robust_evaluate(data_loader_val, model, attacker, device, logger)
            print(f"Robust accuracy of the network on the {len(dataset_val)} test images: {robust_test_stats['acc1']:.1f}%")
            logger.info(f"Robust accuracy of the network on the {len(dataset_val)} test images: {robust_test_stats['acc1']:.1f}%")

            if max_robust_accuracy < robust_test_stats["acc1"]:
                max_robust_accuracy = robust_test_stats["acc1"]

            print(f'Max robust accuracy: {max_robust_accuracy:.2f}%')
            logger.info(f'Max robust accuracy: {max_robust_accuracy:.2f}%')



            log_stats = {'epoch': epoch,
                         **{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         **{f'robust_test_{k}': v for k, v in robust_test_stats.items()},
                         'n_parameters': n_parameters}

            logger.info(log_stats)

            # if args.output_dir and utils.is_main_process():
            #     with (output_dir / "log.txt").open("a") as f:
            #         f.write(json.dumps(log_stats) + "\n")
        else:
            log_stats = {'epoch': epoch,
                         **{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'n_parameters': n_parameters}

            logger.info(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
