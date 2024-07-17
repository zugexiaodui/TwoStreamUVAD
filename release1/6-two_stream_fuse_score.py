import numpy as np
import numpy.lib.npyio
from os.path import join
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
import pickle
from typing import Dict
import colorama

from misc import get_time_stamp, get_logger, get_result_dir, format_args
from metrics import cal_macro_auc, cal_micro_auc
from argmanager import two_stream_fuse_score_parser

TIME_STAMP = get_time_stamp()


def fuse_macro_auc(score_dict_CR: Dict[str, np.ndarray], score_dict_KR: Dict[str, np.ndarray], gt_npz: numpy.lib.npyio.NpzFile, args):
    '''
    Calculate fused macro-AUC.
    '''
    auc_CR, cat_score_dict_CR = cal_macro_auc(score_dict_CR, gt_npz, args.CR_slen, args.CR_sitv, return_score=True)
    auc_KR, cat_score_dict_KR = cal_macro_auc(score_dict_KR, gt_npz, args.KR_slen, args.KR_sitv, return_score=True)

    assert len(cat_score_dict_CR) == len(cat_score_dict_KR) == len(gt_npz)

    fused_score_dict = {}
    auc_dict = {}
    for vid_name in sorted(cat_score_dict_CR.keys()):
        gt_array = gt_npz[vid_name]
        if np.all(gt_array == gt_array[0]):
            gt_array[0] = 1 - gt_array[0]

        score_CR = cat_score_dict_CR[vid_name]
        score_KR = cat_score_dict_KR[vid_name]
        assert score_CR.ndim == score_KR.ndim == 1, f"{score_CR.shape}, {score_KR.shape}"

        fused_score = args.CR_weight * score_CR + args.KR_weight * score_KR
        # fused_score = gaussian_filter1d(fused_score, sigma=args.CR_sitv + args.KR_sitv, axis=0, mode='constant')

        auc_dict[vid_name] = roc_auc_score(gt_array, fused_score)

        fused_score_dict[vid_name] = fused_score

    auc_fused = np.mean(np.array(list(auc_dict.values())))

    ret_dict = {'auc_CR': auc_CR, 'auc_KR': auc_KR, 'auc_fused': auc_fused, 'fused_score_dict': fused_score_dict, 'cat_score_dict_CR': cat_score_dict_CR, 'cat_score_dict_KR': cat_score_dict_KR}
    return ret_dict


def fuse_micro_auc(score_dict_CR: Dict[str, np.ndarray], score_dict_KR: Dict[str, np.ndarray], gt_npz: numpy.lib.npyio.NpzFile, args):
    '''
    Calculate fused micro-AUC.
    '''

    auc_CR, cat_score_CR, cat_gt_CR = cal_micro_auc(score_dict_CR, gt_npz, args.CR_slen, args.CR_sitv, return_score_gt=True)
    auc_KR, cat_score_KR, cat_gt_KR = cal_micro_auc(score_dict_KR, gt_npz, args.KR_slen, args.KR_sitv, return_score_gt=True)

    assert np.all(cat_gt_CR == cat_gt_KR)
    assert cat_score_CR.ndim == cat_score_KR.ndim == 1, f"{cat_score_CR.shape}, {cat_score_KR.shape}"

    fused_cat_score = args.CR_weight * cat_score_CR + args.KR_weight * cat_score_KR
    # fused_cat_score = gaussian_filter1d(fused_cat_score, sigma=args.CR_sitv + args.KR_sitv, axis=0, mode='constant')

    auc_fused = roc_auc_score(cat_gt_CR, fused_cat_score)

    ret_dict = {'auc_CR': auc_CR, 'auc_KR': auc_KR, 'auc_fused': auc_fused, 'fused_cat_score': fused_cat_score, 'cat_score_CR': cat_score_CR, 'cat_score_KR': cat_score_KR}
    return ret_dict


def get_one_channel_score(src_score_dict: Dict[str, np.ndarray], channel: int):
    dst_score_dict = {}
    for vid_name, vid_score in src_score_dict.items():
        if vid_score.ndim == 1:
            vid_score = np.expand_dims(vid_score, -1)
        elif vid_score.ndim != 2:
            raise ValueError(f"`vid_score.ndim` (={vid_score.ndim}) should be 1 or 2.")

        if 0 <= channel < vid_score.shape[1]:
            dst_score_dict[vid_name] = vid_score[:, channel]
        else:
            raise ValueError(f"`channel`={channel} (0 <= channel < {vid_score.shape[1]})")
    return dst_score_dict


if __name__ == "__main__":
    args = two_stream_fuse_score_parser().parse_args()
    logger = get_logger(TIME_STAMP, '' if args.debug_mode else __file__, args.log_root)
    # logger.info(format_args(args))
    if args.debug_mode:
        logger.info(colorama.Fore.RED + colorama.Style.BRIGHT + f"ATTENTION: You are in DEBUG mode. Nothing will be saved!" + colorama.Fore.RESET + colorama.Style.NORMAL)

    gt_npz = np.load(args.gtnpz_path)
    score_dict_CR = np.load(args.score_dict_CR_path)
    score_dict_KR = np.load(args.score_dict_KR_path)

    score_dict_CR_channel = get_one_channel_score(score_dict_CR, args.channel)
    score_dict_KR_channel = get_one_channel_score(score_dict_KR, 0)
    micro_fused_dict = fuse_micro_auc(score_dict_CR_channel, score_dict_KR_channel, gt_npz, args)
    macro_fused_dict = fuse_macro_auc(score_dict_CR_channel, score_dict_KR_channel, gt_npz, args)

    logger.info(f"Fusion of '{args.score_dict_CR_path}'(channel {args.channel}) and '{args.score_dict_KR_path}':")
    # logger.warning(f"Fusion of '{args.score_dict_CR_path}'(channel {args.channel}) and '{args.score_dict_KR_path}':")
    # logger.debug(f"Fusion of '{args.score_dict_CR_path}'(channel {args.channel}) and '{args.score_dict_KR_path}':")

    logger.info(f"Context Reconstruction: \t Micro-AUC={micro_fused_dict['auc_CR']:.2%}, Macro-AUC={macro_fused_dict['auc_CR']:.2%}")
    logger.info(f"Knowledge Retrieval: \t Micro-AUC={micro_fused_dict['auc_KR']:.2%}, Macro-AUC={macro_fused_dict['auc_KR']:.2%}")
    logger.info(f"Two-stream Fusion: \t Micro-AUC={micro_fused_dict['auc_fused']:.2%}, Macro-AUC={macro_fused_dict['auc_fused']:.2%}")

    if not args.debug_mode:
        micro_fused_dict_path = join(get_result_dir(args.result_root), f"score_dict_{TIME_STAMP}_micro_fused.pkl")
        with open(micro_fused_dict_path, 'wb') as f:
            pickle.dump(micro_fused_dict, f)
        logger.info(f"Saved micro_fused_dict to: {micro_fused_dict_path}")

        macro_fused_dict_path = join(get_result_dir(args.result_root), f"score_dict_{TIME_STAMP}_macro_fused.pkl")
        with open(macro_fused_dict_path, 'wb') as f:
            pickle.dump(macro_fused_dict, f)
        logger.info(f"Saved macro_fused_dict to: {macro_fused_dict_path}")
