from collections import OrderedDict
from os import makedirs
from os.path import join, basename, isfile, exists, dirname
from time import time as ttime
from typing import Tuple

import torch
import torch.cuda
import torch.multiprocessing as mp
from torch.backends import cudnn

from stunet import STUNet
from dsets import TestingSetFrame

from misc import get_time_stamp, get_logger, format_args, get_dir_name
from argmanager import feat_stream_KR_parser

TIME_STAMP = get_time_stamp()


def make_dir_check(dst_dir):
    if not exists(dst_dir):
        try:
            makedirs(dst_dir, exist_ok=True)
        except:
            print(f"CANNOT `makedirs '{dst_dir}'`. PLEASE CHECK THE SAVED FILES.")


def load_model(args) -> Tuple[STUNet, str]:
    model = STUNet(args.encoder_path, args.encoder_reinit, args.encoder_rmrelu, args.only_feat)

    if args.resume:
        if isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))

            ckpt_time = str(basename(dirname(args.resume))).split('_')[-1]
            assert len(ckpt_time.split('-')) == 2, f"{ckpt_time}"
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
        print(f"[Checkpoint] Using pretrained model '{args.encoder_path}' without loading a checkpoint.")
        ckpt_time = basename(args.encoder_path).split('.')[0]
    model.eval()
    return model, ckpt_time


def extract_feature(i_proc: int, proc_cnt: int, args):
    '''
    Extract features
    '''
    data_name = str(args.data).split('data.')[1].split('/')[0]
    if not data_name.upper() in ('ST', 'AVENUE', 'CORRIDOR', 'PED2', 'AVE', 'COR'):
        raise NotImplementedError(f"Unknown dataset: {data_name}")

    is_training = 'Train'.upper() in f"{args.data}".upper()
    if not is_training:
        assert 'Test'.upper() in f"{args.data}".upper()

    gpu_id = i_proc % torch.cuda.device_count()

    test_dataset = TestingSetFrame(args.data, args.snippet_len, args.snippet_itv, args.nzfill, args.to_gpu_thres, gpu_id)

    model, ckpt_time = load_model(args)
    model.eval()
    model.cuda(gpu_id)

    if args.print_model and i_proc == 0:
        print(model)

    _save_sfx = '_' + args.save_suffix if args.save_suffix else ''
    _save_sfx.replace(' ', '_')

    flag_start = not bool(args.last_finished)

    for vid_idx in range(i_proc, len(test_dataset), proc_cnt):
        vid_name = test_dataset.vid_name_list[vid_idx]

        if not flag_start:
            if proc_cnt > 1:
                raise NotImplementedError("`last_finished` has not been implemented for more than one processes.")
            if vid_name == args.last_finished:
                flag_start = True
            print(f"Skip {vid_name}")
            continue

        vid_name, vid_stack = test_dataset[vid_idx]

        print(f"({vid_idx+1}/{len(test_dataset)}): {vid_name}, {vid_stack.shape[2]} frames")
        test_feat_dict = OrderedDict()

        n_snippets = test_dataset.sta_frm_dict[vid_name] + 1

        for _snippet_idx in range(n_snippets):
            frm_idx_list = test_dataset.sample_frms_idx(_snippet_idx)
            frm_stack = vid_stack[:, :, torch.as_tensor(frm_idx_list), :, :]

            with torch.no_grad():
                if frm_stack.device.type == 'cpu':
                    frm_stack = frm_stack.cuda(gpu_id, non_blocking=True)

                feat = model(frm_stack)
                feat: torch.Tensor = feat.detach().cpu()
                feat = feat.permute((0, 2, 3, 4, 1))  # (N, C, T, H, W) -> (N, T, H, W, C)

            if is_training:
                dst_dir = f"../{args.feat_root}/{data_name}/{get_dir_name(__file__)}_{ckpt_time}{_save_sfx}/Train/{vid_name}"
                make_dir_check(dst_dir)
                torch.save(feat, join(dst_dir, f"{str(_snippet_idx).zfill(6)}.pth"))
            else:
                test_feat_dict[_snippet_idx] = feat

        if not is_training:
            dst_dir = f"../{args.feat_root}/{data_name}/{get_dir_name(__file__)}_{ckpt_time}{_save_sfx}/Test"
            make_dir_check(dst_dir)
            torch.save(test_feat_dict, join(dst_dir, f"{vid_name}.pth"))
        print(f"Finished {vid_name}")


if __name__ == '__main__':
    args = feat_stream_KR_parser().parse_args()

    logger = get_logger(TIME_STAMP, '' if args.debug_mode else __file__, args.log_root)
    logger.info(format_args(args))
    if args.debug_mode:
        logger.info(f"ATTENTION: You are in DEBUG mode. Nothing will be saved!")

    cudnn.benchmark = not args.debug_mode
    torch.set_num_threads(args.threads)

    logger.info(f"Extracting feature ...")
    t0 = ttime()

    mp.spawn(extract_feature, args=(args.workers, args), nprocs=args.workers)

    t1 = ttime()
    logger.info(f"Time={(t1-t0)/60:.1f} min")
