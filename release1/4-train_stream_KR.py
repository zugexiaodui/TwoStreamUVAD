import torch
import random
import numpy as np
from functools import partial

rand_seed = 2021
random.seed(rand_seed)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)
np.random.seed(rand_seed)

import torch.nn as nn
import torch.utils.data

from hashnet import HashNet, mutual_loss
from dsets import TrainingSetFeature

from misc import get_time_stamp, get_logger, format_args, get_ckpt_dir, save_checkpoint
from argmanager import train_stream_KR_parser


TIME_STAMP = get_time_stamp()


if __name__ == '__main__':
    args = train_stream_KR_parser().parse_args()

    logger = get_logger(TIME_STAMP, '' if args.debug_mode else __file__, args.log_root)
    logger.info(format_args(args))
    if args.debug_mode:
        logger.info(f"ATTENTION: You are in DEBUG mode. Nothing will be saved!")

    train_dataset = TrainingSetFeature(args.train_data, args.t_rand_range, args.iterations)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=False, prefetch_factor=1)

    model = HashNet(feat_dim=[1, 2, 2, 2048], len_hash_code=args.len_hash_code, num_hash_layer=args.num_hash_layer)
    model.train()
    model = model.cuda()

    if args.print_model:
        logger.info(f"{model}")

    criterion = nn.CosineSimilarity(dim=-1)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if not args.debug_mode:
        ckpt_save_func = partial(save_checkpoint, is_best=False, filedir=get_ckpt_dir(TIME_STAMP, __file__, args.ckpt_root), writer=logger.info)
        ckpt_save_func({'epoch': 0,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       epoch=0)

    lam_m: float = args.lam_m
    _b: int = args.num_hash_layer
    _r: int = args.len_hash_code

    for epoch in range(args.epochs):
        for step, snippets in enumerate(train_loader):
            feat0: torch.Tensor = snippets[0].cuda(non_blocking=True)
            feat1: torch.Tensor = snippets[1].cuda(non_blocking=True)

            hcode0: torch.Tensor = model(feat0)
            hcode1: torch.Tensor = model(feat1)

            loss_c = 1 - torch.mean(criterion(hcode0, hcode1))
            _loss_m0 = mutual_loss(hcode0.reshape(hcode0.shape[0], _b, _r))
            _loss_m1 = mutual_loss(hcode1.reshape(hcode1.shape[0], _b, _r))
            loss_m = (_loss_m0 + _loss_m1) * 0.5
            loss_all = loss_c + lam_m * loss_m

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            if step % args.print_freq == 0:
                logger.info(f"Epoch {epoch} Step {step}: loss_cos={loss_c.item():.4f} " +
                            f"loss_mut={lam_m}*{loss_m.item():.4f} " +
                            f"loss_all={loss_all.item():.4f}")

        if not args.debug_mode:
            if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
                ckpt_save_func({'epoch': epoch + 1,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict()},
                               epoch=epoch + 1)
