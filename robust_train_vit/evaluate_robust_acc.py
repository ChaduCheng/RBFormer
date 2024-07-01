
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
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate, robust_evaluate
from losses import DistillationLoss
from model_adv_att import AttackPGD
from samplers import RASampler
from vit_pytorch.Robust_VITmodel import Robust_ViTmodel
from autoattack import AutoAttack
# from vit_pytorch.model.vision_transformer import vit_b16_224
from vit_pytorch.vision_transformer import vit_b16_224

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=1000, type=int)
    parser.add_argument('--epochs', default=600, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_tiny_patch4_32', type=str, metavar='MODEL',
                        help='Name of model to train')
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
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
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
    parser.add_argument('--data-path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'TinyIMN', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./results/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=' ', help='resume from checkpoint')
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
    parser.add_argument('--advtrain', default=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    ### model elements


    parser.add_argument('--patch', default='4', type=int)

    parser.add_argument('--embedding', default='conv_input', help='which embedding using here')
    parser.add_argument('--Transblock', default='MLP_att', help='which embedding using here')
    parser.add_argument('--MLP', default='conv_MLP', help='which embedding using here')
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


    ### dataset

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


    model = vit_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True,
                                   representation_size=768).to(device)
    print(model)


    output_dir_all = args.output_dir + args.embedding + '-' + args.Transblock + '-' + args.MLP + '-' +  args.norm + '-' + args.skipconnect + '/'

    output_dir = Path(output_dir_all)

    if output_dir_all:
        output_dir.mkdir(parents=True, exist_ok=True)


    log_file = output_dir_all + 'robust_acc.log'

    logger = get_logger(log_file)


    logger.info('model structure')


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
                print(f'Removing key {k} from pretrained checkpoint')
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

    # model.to(device)

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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

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
        print(f'Creating teacher model: {args.teacher_model}')
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

    
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    resume_dir_all = args.resume + args.embedding + '-' + args.Transblock + '-' + args.MLP + '-' +  args.norm + '-' + args.skipconnect + '/checkpoint.pth'

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                resume_dir_all, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            # print(checkpoint)
        # model_without_ddp.load_state_dict(checkpoint['model'].module)
        model_without_ddp.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
        # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #     args.start_epoch = checkpoint['epoch'] + 1
        #     if args.model_ema:
        #         utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
        #     if 'scaler' in checkpoint:
        #         loss_scaler.load_state_dict(checkpoint['scaler'])

    # Initialize attacker
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    mu = torch.tensor(cifar10_mean).view(3, 1, 1).to(device)
    std = torch.tensor(cifar10_std).view(3, 1, 1).to(device)


    config_linf = {
        'epsilon': torch.tensor(8.0 / 255),
        'num_steps': 10,
        # 'step_size': 2.0 / 255,
        'step_size': 0.01,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'linf'
    }

    config_linf2 = {
        'epsilon': torch.tensor(2.0 / 255),
        'num_steps': 10,
        # 'step_size': 2.0 / 255,
        'step_size': 0.01,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'linf'
    }

    config_linf3 = {
        'epsilon': torch.tensor(3.0 / 255),
        'num_steps': 10,
        # 'step_size': 2.0 / 255,
        'step_size': 0.01,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'linf'
    }

    config_linf4 = {
        'epsilon': torch.tensor(4.0 / 255),
        'num_steps': 10,
        # 'step_size': 2.0 / 255,
        'step_size': 0.01,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'linf'
    }


    # print(dataset_val.transforms.transform)
    # sys.exit()

    model = NormWrapper(mu, std, model)


    if args.eval:
        # clean evaluation
        logger.info('Evaluating Clean Accuracy')
        test_stats = evaluate(data_loader_val, model, device, logger)
        print(f"Clean Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        logger.info(f"Clean Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        # pgd evaluation
        # logger.info('Evaluating PGD Robust Accuracy')
        pgd_attacker1 = AttackPGD(model, config=config_linf).to(device)
        pgd_test_stats1 = robust_evaluate(data_loader_val, model, pgd_attacker1, device, logger)
        print(f"PGD 1: Robust accuracy of the network on the {len(dataset_val)} test images: {pgd_test_stats1['acc1']:.1f}%")
        logger.info(f"PGD 1: Robust accuracy of the network on the {len(dataset_val)} test images: {pgd_test_stats1['acc1']:.1f}%")

        # # pgd evaluation
        # # logger.info('Evaluating PGD Robust Accuracy')
        # pgd_attacker2 = AttackPGD(model, config=config_linf2).to(device)
        # pgd_test_stats2 = robust_evaluate(data_loader_val, model, pgd_attacker2, device, logger)
        # print(f"PGD 2: Robust accuracy of the network on the {len(dataset_val)} test images: {pgd_test_stats2['acc1']:.1f}%")
        # logger.info(f"PGD 2: Robust accuracy of the network on the {len(dataset_val)} test images: {pgd_test_stats2['acc1']:.1f}%")
        #
        # # pgd evaluation
        # # logger.info('Evaluating PGD Robust Accuracy')
        # pgd_attacker3 = AttackPGD(model, config=config_linf3).to(device)
        # pgd_test_stats3 = robust_evaluate(data_loader_val, model, pgd_attacker3, device, logger)
        # print(f"PGD 3: Robust accuracy of the network on the {len(dataset_val)} test images: {pgd_test_stats3['acc1']:.1f}%")
        # logger.info(f"PGD 3: Robust accuracy of the network on the {len(dataset_val)} test images: {pgd_test_stats3['acc1']:.1f}%")
        #
        # # pgd evaluation
        # # logger.info('Evaluating PGD Robust Accuracy')
        # pgd_attacker4 = AttackPGD(model, config=config_linf4).to(device)
        # pgd_test_stats4 = robust_evaluate(data_loader_val, model, pgd_attacker4, device, logger)
        # print(f"PGD 4: Robust accuracy of the network on the {len(dataset_val)} test images: {pgd_test_stats4['acc1']:.1f}%")
        # logger.info(f"PGD 4: Robust accuracy of the network on the {len(dataset_val)} test images: {pgd_test_stats4['acc1']:.1f}%")


        # auto-attack evaluation
        auto_attacker1 = AutoAttack(model, norm='Linf', eps=config_linf['epsilon'], is_tf_model=False)
        logger.info('Evaluating AutoAttack Robust Accuracy')
        autoattack_test_stats1 = robust_evaluate(data_loader_val, model, auto_attacker1, device, logger)
        print(f"AutoAttack 1: Robust accuracy of the network on the {len(dataset_val)} test images: {autoattack_test_stats1['acc1']:.1f}%")
        logger.info(f"AutoAttack 1: Robust accuracy of the network on the {len(dataset_val)} test images: {autoattack_test_stats1['acc1']:.1f}%")


        # auto_attacker2 = AutoAttack(model, norm='Linf', eps=config_linf2['epsilon'], version='plus', is_tf_model=False)
        # logger.info('Evaluating AutoAttack Robust Accuracy')
        # autoattack_test_stats2 = robust_evaluate(data_loader_val, model, auto_attacker2, device, logger)
        # print(f"AutoAttack 2: Robust accuracy of the network on the {len(dataset_val)} test images: {autoattack_test_stats2['acc1']:.1f}%")
        # logger.info(f"AutoAttack 2: Robust accuracy of the network on the {len(dataset_val)} test images: {autoattack_test_stats2['acc1']:.1f}%")
        #


        # auto_attacker3 = AutoAttack(model, norm='Linf', eps=config_linf3['epsilon'], version='plus', is_tf_model=False)
        # logger.info('Evaluating AutoAttack Robust Accuracy')
        # autoattack_test_stats3 = robust_evaluate(data_loader_val, model, auto_attacker3, device, logger)
        # print(f"AutoAttack 3: Robust accuracy of the network on the {len(dataset_val)} test images: {autoattack_test_stats3['acc1']:.1f}%")
        #


        # print(f"AutoAttack 1: Robust accuracy of the network on the {len(dataset_val)} test images: {autoattack_test_stats1['acc1']:.1f}%")
        # logger.info(f"AutoAttack 1: Robust accuracy of the network on the {len(dataset_val)} test images: {autoattack_test_stats1['acc1']:.1f}%")
        #
        # print(f"AutoAttack 2: Robust accuracy of the network on the {len(dataset_val)} test images: {autoattack_test_stats2['acc1']:.1f}%")
        # logger.info(f"AutoAttack 2: Robust accuracy of the network on the {len(dataset_val)} test images: {autoattack_test_stats2['acc1']:.1f}%")

        # print(f"AutoAttack 3: Robust accuracy of the network on the {len(dataset_val)} test images: {autoattack_test_stats3['acc1']:.1f}%")
        # logger.info(f"AutoAttack 3: Robust accuracy of the network on the {len(dataset_val)} test images: {autoattack_test_stats3['acc1']:.1f}%")

        # auto_attacker4 = AutoAttack(model, norm='Linf', eps=config_linf4['epsilon'], version='plus', is_tf_model=False)
        # logger.info('Evaluating AutoAttack Robust Accuracy')
        # autoattack_test_stats4 = robust_evaluate(data_loader_val, model, auto_attacker4, device, logger)
        # print(f"AutoAttack 4: Robust accuracy of the network on the {len(dataset_val)} test images: {autoattack_test_stats4['acc1']:.1f}%")
        # logger.info(f"AutoAttack 4: Robust accuracy of the network on the {len(dataset_val)} test images: {autoattack_test_stats4['acc1']:.1f}%")
        #

        return

class NormWrapper(torch.nn.Module):

    def __init__(self, mu, std, model):
        super(NormWrapper, self).__init__()

        self.mu = mu
        self.std = std
        self.model = model

    def forward(self, x):
        x = (x - self.mu) / self.std
        return self.model(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
