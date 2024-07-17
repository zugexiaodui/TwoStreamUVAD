import numpy as np
import tqdm
from os import listdir
from os.path import join, isfile, exists
from time import time as ttime
from typing import Union, Tuple

import torch
import torch.cuda
import torch.multiprocessing as mp

from hashnet import HashNet
from dsets import TestingSetFeature
from il2sh import iL2SH

from misc import get_time_stamp, get_logger, get_result_dir, format_args
from metrics import cal_macro_auc, cal_micro_auc
from argmanager import test_stream_KR_parser

TIME_STAMP = get_time_stamp()


def load_model(args) -> Tuple[HashNet, int]:
    model = HashNet(feat_dim=[1, 2, 2, 2048], len_hash_code=args.len_hash_code, num_hash_layer=args.num_hash_layer)

    if args.resume:
        if isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch: int = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(args.resume))
    else:
        raise NotImplementedError("A checkpoint should be loaded.")

    model.eval()
    return model, epoch


def cal_anomaly_score(i_proc: int, proc_cnt: int, score_queue: mp.Queue, args, il2sh_path: str):
    '''
    Calculate anomaly scores 
    '''
    il2sh_inst: iL2SH = torch.load(il2sh_path)

    test_dataset = TestingSetFeature(root_dir=args.test_data)

    for vid_idx in range(i_proc, len(test_dataset), proc_cnt):
        vid_name, vid_data = test_dataset[vid_idx]

        n_snippets = len(vid_data)
        print(f"({vid_idx+1}/{len(test_dataset)}): \t{vid_name}, {n_snippets} snippets")

        vid_scores: np.ndarray = np.zeros(n_snippets)

        score_dict = {}
        for _snippet_idx in range(n_snippets):
            _snippet_data: torch.Tensor = vid_data[_snippet_idx]  # [Ncrop, T, H, W, C]

            _candist: torch.Tensor = il2sh_inst.batch_retrieval(_snippet_data)  # [Ncrop, b]
            _snippet_score = _candist.amin(1).mean()
            vid_scores[_snippet_idx] = _snippet_score.item()

        if 'ave' in args.test_data.lower() and n_snippets == 0:
            print(f"Video '{vid_name}' is too short. All frames will be asigned the same score (0).")
            vid_scores = np.array([0.] * 35 + [0.00001])

        score_dict[vid_name] = vid_scores

        assert not score_queue.full()
        score_queue.put(score_dict)


if __name__ == '__main__':
    args = test_stream_KR_parser().parse_args()

    logger = get_logger(TIME_STAMP, '' if args.debug_mode else __file__, args.log_root)
    logger.info(format_args(args))
    if args.debug_mode:
        logger.info(f"ATTENTION: You are in DEBUG mode. Nothing will be saved!")

    torch.set_num_threads(args.threads)

    t0 = ttime()

    gt_npz = np.load(args.gtnpz_path)

    if args.score_dict_path:
        logger.info(f"Using the score_dict from '{args.score_dict_path}'")
        assert exists(args.score_dict_path), f"{args.score_dict_path}"
        score_dict = np.load(args.score_dict_path)
    else:
        if torch.cuda.device_count() != 1:
            logger.warn("Please use only one GPU.")
            exit()

        if not args.il2sh_inst_path:
            if args.debug_mode:
                if input("Something (iL2SH instance) must be saved in the following steps. Continue? (y/n)") != 'y':
                    exit()

            # Load model
            _hash_net, epoch = load_model(args)

            il2sh_inst = iL2SH(_hash_net)

            # Index stage
            logger.info("Constructing hash tables ...")
            for _vid_name in tqdm.tqdm(sorted(listdir(args.train_data)), desc='Construction'):
                vid_dir = join(args.train_data, _vid_name)
                train_mat = []
                for _snippet_name in sorted(listdir(vid_dir)):
                    try:
                        snippet_pth: torch.Tensor = torch.load(join(vid_dir, _snippet_name))
                    except:
                        print(join(vid_dir, _snippet_name))
                    # train_mat.append(snippet_pth.flatten(1))
                    train_mat.append(snippet_pth)
                train_mat = torch.cat(train_mat, 0)
                il2sh_inst.batch_construction(train_mat)

            # Save the model temporarily
            il2sh_path = join(get_result_dir(args.result_root), f"il2sh_inst_{TIME_STAMP}_{epoch}.pth")
            torch.save(il2sh_inst, il2sh_path)
            logger.info(f"Saved il2sh_inst to: {il2sh_path}")
            il2sh_inst.show_tables()

            del il2sh_inst
            torch.cuda.empty_cache()
        else:
            logger.info(f"Using the il2sh_inst from '{args.il2sh_inst_path}'")
            il2sh_path = args.il2sh_inst_path

        # Query stage
        logger.info("Retrieving from hash tables  ...")
        len_dataset = len(TestingSetFeature(root_dir=args.test_data))
        score_queue = mp.Manager().Queue(maxsize=len_dataset)

        mp.spawn(cal_anomaly_score, args=(args.workers, score_queue, args, il2sh_path), nprocs=args.workers)

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

    # Calculate AUCs
    macro_auc = cal_macro_auc(score_dict, gt_npz, args.snippet_len, args.snippet_itv)
    micro_auc = cal_micro_auc(score_dict, gt_npz, args.snippet_len, args.snippet_itv)
    logger.info(f"Micro-AUC = {micro_auc:.2%}, Macro-AUC = {macro_auc:.2%}")

    t1 = ttime()
    logger.info(f"Time={(t1-t0)/60:.1f} min")
