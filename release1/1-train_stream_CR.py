import os
from functools import partial

import torch
import random
import numpy as np

rand_seed = 2021
random.seed(rand_seed)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)
np.random.seed(rand_seed)

import torch.nn as nn
import torch.utils.data
from torch.backends import cudnn

from stunet import STUNet
from dsets import TrainingSetFrame

from misc import get_time_stamp, get_logger, format_args, get_ckpt_dir, save_checkpoint
from argmanager import train_stream_CR_parser

from tensorboardX import SummaryWriter


TIME_STAMP = get_time_stamp()


if __name__ == '__main__':
    args = train_stream_CR_parser().parse_args()

    logger = get_logger(TIME_STAMP, '' if args.debug_mode else __file__, args.log_root)
    logger.info(format_args(args))
    if args.debug_mode:
        logger.info(f"ATTENTION: You are in DEBUG mode. Nothing will be saved!")

    cudnn.benchmark = not args.debug_mode

    n_gpus = torch.cuda.device_count()

    train_dataset = TrainingSetFrame(args.data, args.snippet_len, args.snippet_itv, args.nzfill, args.iterations, args.fixed_3crop)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=False, prefetch_factor=1)

    model = STUNet(args.encoder_path, args.encoder_reinit, args.encoder_rmrelu, args.only_feat)
    model = torch.nn.parallel.DataParallel(model)
    model.train()
    model = model.cuda()

    if args.print_model:
        logger.info(f"{model}")

    if args.fr == 0:
        # 冻结encoder参数，BN层统计参数仍然更新
        for n, p in model.named_parameters():
            if 'encoder' in n:
                p.requires_grad_(False)

        # 收集可更新参数
        logger.info("[Learning Rate] Set requires_grad ...")
        param_list = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                param_list.append(p)
            else:
                logger.info(f"{n}: requires_grad={p.requires_grad}")

    elif args.fr == 1:
        logger.info("[Learning Rate] All layers have the same `lr` ")
        param_list = []
        for n, p in model.named_parameters():
            param_list.append(p)

    elif args.fr > 0:
        # # 调小ecnoder的BN层动量
        # logger.info("[BatchNorm3D] Decay the `BatchNorm3d.momentum` ...")
        # for n, m in model.named_modules():
        #     if 'encoder' in n:
        #         if isinstance(m, nn.BatchNorm3d):
        #             m.momentum = 0.1 * m.momentum

        # 收集所有可更新参数，并设置相应的学习倍率
        param_list = [{'params': [], 'lr':args.lr},
                      {'params': [], 'lr':args.lr * args.fr}]
        pname_list = {_p['lr']: [] for _p in param_list}

        logger.info("[Learning Rate] Set finetuning_rate ...")
        for n, p in model.named_parameters():
            if p.requires_grad:
                _group_idx = 1 if 'encoder' in n else 0
                param_list[_group_idx]['params'].append(p)
                pname_list[param_list[_group_idx]['lr']].append(n)
            else:
                logger.info(f"{n}: requires_grad={p.requires_grad}")

        for _lr, _pn in pname_list.items():
            logger.info(f'[Optimizer] lr={_lr}:')
            logger.info(' | '.join(_pn))

    else:
        raise ValueError(f"`args.fr({args.fr})` shoule be >=0")

    criterion_mse = nn.MSELoss().cuda()
    criterion_l1 = nn.L1Loss().cuda()

    optimizer = torch.optim.Adam(param_list, args.lr)
    lr_sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=0.1)

    if not args.debug_mode:
        ckpt_save_func = partial(save_checkpoint, is_best=False, filedir=get_ckpt_dir(TIME_STAMP, __file__, args.ckpt_root), writer=logger.info)
        tbxs_witer = SummaryWriter(get_ckpt_dir(TIME_STAMP, __file__, args.tbxs_root))

    start_epoch = 1
    if args.resume:
        if os.path.exists(args.resume) and os.path.isfile(args.resume):
            ckpt = torch.load(args.resume)
            model.module.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_sch.load_state_dict(ckpt['lr_sch'])
            start_epoch = ckpt['epoch'] + 1
        else:
            raise FileNotFoundError(f"{args.resume}")
    else:
        if not args.debug_mode:
            ckpt_save_func({'epoch': 0,
                            'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_sch': lr_sch.state_dict()},
                           epoch=0)

    for epoch in range(start_epoch, args.epochs + 1):
        _sumloss_mse = 0
        _sumloss_l1 = 0
        _sumloss_all = 0

        for step, (inputs, target) in enumerate(train_loader):
            inputs: torch.Tensor = inputs.cuda(non_blocking=True)
            target: torch.Tensor = target.cuda(non_blocking=True)

            output = model(inputs)

            loss_mse: torch.Tensor = criterion_mse(output, target)
            loss_l1: torch.Tensor = criterion_l1(output, target)
            loss_all: torch.Tensor = loss_mse + args.lam_l1 * loss_l1

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            if step % args.print_freq == 0:
                logger.info(f"Epoch[{epoch}/{args.epochs}] step {step:>3d}/{len(train_loader)}: " +
                            f"lr={lr_sch.get_last_lr()[0]} " +
                            f"loss_mse={loss_mse.item():.4f} loss_l1={args.lam_l1}*{loss_l1.item():.4f} " +
                            f"loss_all={loss_all.item():.4f}")

            if not args.debug_mode:
                _global_step = (epoch - 1) * len(train_loader) + step
                tbxs_witer.add_scalar('step/lr', lr_sch.get_last_lr()[0], _global_step)
                tbxs_witer.add_scalar('step/loss_mse', loss_mse.item(), _global_step)
                tbxs_witer.add_scalar('step/loss_l1', loss_l1.item(), _global_step)
                tbxs_witer.add_scalar('step/loss_all', loss_all.item(), _global_step)

                _sumloss_mse += len(target) * loss_mse.item()
                _sumloss_l1 += len(target) * loss_l1.item()
                _sumloss_all += len(target) * loss_all.item()

        if not args.debug_mode:
            tbxs_witer.add_scalar('epoch/lr', lr_sch.get_last_lr()[0], epoch)
            tbxs_witer.add_scalar('epoch/loss_mse', _sumloss_mse / len(train_dataset), epoch)
            tbxs_witer.add_scalar('epoch/loss_l1', _sumloss_l1 / len(train_dataset), epoch)
            tbxs_witer.add_scalar('epoch/loss_all', _sumloss_all / len(train_dataset), epoch)

            if epoch % args.save_freq == 0 or epoch == args.epochs:
                ckpt_save_func({'epoch': epoch,
                                'model': model.module.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_sch': lr_sch.state_dict()},
                               epoch=epoch)

        _last_lr = lr_sch.get_last_lr()[0]
        lr_sch.step()
        if lr_sch.get_last_lr()[0] < _last_lr:
            logger.info(f"[Learning Rate] Decay `lr` from {_last_lr} to {lr_sch.get_last_lr()[0]}")
