import numpy as np
from collections import OrderedDict
from os.path import join, exists, isfile
from time import time as ttime
from typing import Tuple
import functools

import torch
import torch.cuda
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.backends import cudnn

from stunet import STUNet
from dsets import TestingSetFrame, TestingSetVideo

from misc import get_time_stamp, get_logger, get_result_dir, format_args
from metrics import cal_macro_auc, cal_micro_auc
from argmanager import test_stream_CR_parser


TIME_STAMP = get_time_stamp()


def load_model(args) -> STUNet:
    model = STUNet(args.encoder_path, args.encoder_reinit, args.encoder_rmrelu, args.only_feat)

    if args.resume:
        if isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume, map_location='cpu')

            new_state_dict = OrderedDict()
            _prefix = 'module.'
            for k, v in checkpoint['model'].items():
                if k.startswith(_prefix):
                    new_state_dict[k[len(_prefix):]] = v
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict)

        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(args.resume))
    else:
        raise NotImplementedError("A checkpoint should be loaded.")

    model.eval()
    return model


def maximum_local_error(img_true: torch.Tensor, img_test: torch.Tensor, patch_func_list: nn.ModuleList, lam_l1: float, use_channel_l2: bool):
    # [C,H,W]
    assert len(img_true.shape) == len(img_test.shape) == 3, f"{img_true.shape}, {img_test.shape}"
    assert img_true.shape == img_test.shape
    assert img_true.shape[0] == img_test.shape[0] == 3

    if use_channel_l2:
        diff_mse = torch.square(img_true - img_test).sum(0, True).sqrt().div(img_true.shape[0])
    else:
        diff_mse = torch.square(img_true - img_test).mean(0, True)
    diff_l1 = torch.abs(img_true - img_test).mean(0, True)

    patch_score_list = []
    for _patch_func in patch_func_list:
        _patch_err_mse: torch.Tensor = _patch_func(diff_mse)
        _patch_err_l1: torch.Tensor = _patch_func(diff_l1)
        _patch_score = _patch_err_mse.amax() + lam_l1 * _patch_err_l1.amax()
        patch_score_list.append(_patch_score)
    return patch_score_list


def frame_level_error(image_true: torch.Tensor, image_test: torch.Tensor, lam_l1: float):
    return F.mse_loss(image_test, image_true).sqrt() + lam_l1 * F.l1_loss(image_test, image_true)


def cal_anomaly_score(i_proc: int, proc_cnt: int, score_queue: mp.Queue, args):
    '''
    Calculate anomaly scores 
    '''
    gpu_id = i_proc % torch.cuda.device_count()

    if args.vid_dir:
        test_dataset = TestingSetVideo(args.data, args.snippet_len, args.snippet_itv, args.nzfill, args.to_gpu_thres, gpu_id, args.vid_dir)
    else:
        test_dataset = TestingSetFrame(args.data, args.snippet_len, args.snippet_itv, args.nzfill, args.to_gpu_thres, gpu_id)

    model = load_model(args)
    model.cuda(gpu_id)

    if args.print_model and i_proc == 0:
        print(model)

    fuse_func = torch.mean if args.crop_fuse_type == 'mean' else torch.amax

    if args.error_type == 'MLE':
        _avg_pool_list = nn.ModuleList()
        for _patch_size in args.patch_size:
            assert 0 < _patch_size <= 256, f"{_patch_size}"
            _avg_pool_list.append(nn.AvgPool2d(_patch_size, args.patch_stride))
        error_func = functools.partial(maximum_local_error, patch_func_list=_avg_pool_list, lam_l1=args.lam_l1, use_channel_l2=args.use_channel_l2)
    elif args.error_type == 'FLE':
        error_func = functools.partial(frame_level_error, lam_l1=args.lam_l1)
    else:
        raise NameError(f"ERROR args.error_type: {args.error_type}")

    for vid_idx in range(i_proc, len(test_dataset), proc_cnt):
        # vid_name = test_dataset.vid_name_list[vid_idx]
        vid_name, vid_stack = test_dataset[vid_idx]

        print(f"({vid_idx+1}/{len(test_dataset)}): {vid_name} {vid_stack.shape[2]} frames")

        n_snippets = test_dataset.sta_frm_dict[vid_name] + 1
        vid_scores: np.ndarray = np.zeros([n_snippets, len(args.patch_size)])

        score_dict = {}
        for _snippet_idx in range(n_snippets):
            frm_idx_list = test_dataset.sample_frms_idx(_snippet_idx)
            frm_stack = vid_stack[:, :, torch.as_tensor(frm_idx_list), :, :]

            with torch.no_grad():
                if frm_stack.device.type == 'cpu':
                    frm_stack = frm_stack.cuda(gpu_id)

                pred_frm: torch.Tensor = model(frm_stack[:, :, :-1, :, :])

                _true_frm = frm_stack[:, :, -1, :, :]
                _pred_frm = pred_frm.squeeze()
                _snippet_score = fuse_func(torch.as_tensor([error_func(_true_frm[i_crop], _pred_frm[i_crop]) for i_crop in range(3)]), dim=0)

            vid_scores[_snippet_idx] = _snippet_score.numpy()

        score_dict[vid_name] = vid_scores

        assert not score_queue.full()
        score_queue.put(score_dict)


if __name__ == '__main__':
    args = test_stream_CR_parser().parse_args()

    logger = get_logger(TIME_STAMP, '' if args.debug_mode else __file__, args.log_root)
    logger.info(format_args(args))
    if args.debug_mode:
        logger.info(f"ATTENTION: You are in DEBUG mode. Nothing will be saved!")

    cudnn.benchmark = not args.debug_mode
    torch.set_num_threads(args.threads)

    t0 = ttime()

    gt_npz = np.load(args.gtnpz_path)

    if args.score_dict_path:
        logger.info(f"Using the score_dict from '{args.score_dict_path}'")
        assert exists(args.score_dict_path), f"{args.score_dict_path}"
        score_dict = np.load(args.score_dict_path)
    else:
        epoch = torch.load(args.resume, map_location='cpu')['epoch']

        logger.info(f"Testing epoch [{epoch}] ...")
        len_dataset = len(TestingSetFrame(args.data, args.snippet_len, args.snippet_itv, args.nzfill))
        score_queue = mp.Manager().Queue(maxsize=len_dataset)

        mp.spawn(cal_anomaly_score, args=(args.workers, score_queue, args), nprocs=args.workers)

        assert score_queue.full()
        score_dict = {}
        while not score_queue.empty():
            score_dict.update(score_queue.get())
        assert len(score_dict) == len_dataset

        # Save scores
        if not args.debug_mode:
            score_dict_path = join(get_result_dir(args.result_root), f"score_dict_{TIME_STAMP}_{epoch}.npz")
            np.savez(score_dict_path, **score_dict)
            logger.info(f"Saved score_dict to: {score_dict_path}")

    # Calculate AUC
    for _i_patch, _patch_size in enumerate(args.patch_size):
        _p_score_dict = {}
        for _vid_name, _vid_score in score_dict.items():
            _p_score_dict[_vid_name] = _vid_score[:, _i_patch]

        macro_auc = cal_macro_auc(_p_score_dict, gt_npz, args.snippet_len, args.snippet_itv)
        micro_auc = cal_micro_auc(_p_score_dict, gt_npz, args.snippet_len, args.snippet_itv)
        logger.info(f"Patch_size {_patch_size:>3}: Micro-AUC = {micro_auc:.2%}, Macro-AUC = {macro_auc:.2%}")

    t1 = ttime()
    logger.info(f"Time={(t1-t0)/60:.1f} min")
