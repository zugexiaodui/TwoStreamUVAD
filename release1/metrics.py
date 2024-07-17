import numpy as np
import numpy.lib.npyio
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from sklearn.metrics import roc_auc_score
import functools
from typing import Union, Tuple, Dict

gaussian_filter1d = functools.partial(gaussian_filter1d, axis=0, mode='constant')


def cal_macro_auc(score_dict: Dict[str, np.ndarray], gt_npz: numpy.lib.npyio.NpzFile, slen: int, sitv: int, return_score: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    '''
    Calculate macro-AUC.
    '''

    # vid_name_list = sorted(list(gt_npz.keys()))
    vid_name_list = sorted(list(score_dict.keys()))
    cat_score_dict = {}

    auc_dict = {}
    for vid_name in vid_name_list:
        _vid_score = score_dict[vid_name]

        vid_gt: np.ndarray = gt_npz[vid_name]
        if np.all(vid_gt == vid_gt[0]):
            vid_gt[0] = 1 - vid_gt[0]

        assert vid_gt.ndim == _vid_score.ndim == 1, f"{vid_gt.shape}, {_vid_score.shape}"

        _vid_score = normalize_score(_vid_score, 'minmax')
        if not len(vid_gt) == len(_vid_score):
            _vid_score = align_score(slen, sitv, _vid_score)
        _vid_score = gaussian_filter1d(_vid_score, sigma=sitv * slen / 2)

        assert len(vid_gt) == len(_vid_score), f"{vid_name}: {vid_gt.shape}, {_vid_score.shape}"
        auc_dict[vid_name] = roc_auc_score(vid_gt, _vid_score)
        cat_score_dict[vid_name] = _vid_score

    macro_auc = np.mean(np.array(list(auc_dict.values())))

    if return_score:
        return macro_auc, cat_score_dict
    return macro_auc


def cal_micro_auc(score_dict: Dict[str, np.ndarray], gt_npz: numpy.lib.npyio.NpzFile, slen: int, sitv: int, return_score_gt: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    '''
    Calculate micro-AUC.
    '''
    vid_name_list = sorted(list(score_dict.keys()))

    def _concat_score(_score_dict: dict):
        _cat_gts = []
        _cat_scores = []
        for _vid_name in vid_name_list:
            _vid_gt = gt_npz[_vid_name]
            _vid_score: np.ndarray = _score_dict[_vid_name]
            _vid_score = _vid_score.squeeze()
            assert _vid_gt.ndim == _vid_score.ndim == 1, f"{_vid_gt.shape}, {_vid_score.shape}"

            _vid_score = normalize_score(_vid_score, 'minmax')
            if not len(_vid_gt) == len(_vid_score):
                _vid_score = align_score(slen, sitv, _vid_score)
            _vid_score = gaussian_filter1d(_vid_score, sigma=sitv * slen / 2)

            assert len(_vid_gt) == len(_vid_score), f"{_vid_gt.shape}, {_vid_score.shape}, {slen}, {sitv}"
            _cat_gts.append(_vid_gt)
            _cat_scores.append(_vid_score)
        _cat_gts = np.concatenate(_cat_gts)
        _cat_scores = np.concatenate(_cat_scores)
        return _cat_gts, _cat_scores

    cat_gts, cat_scores = _concat_score(score_dict)
    micro_auc = roc_auc_score(cat_gts, cat_scores)

    if return_score_gt:
        return micro_auc, cat_scores, cat_gts
    return micro_auc


def align_score(slen: int, sitv: int, input_score: np.ndarray):
    assert input_score.ndim in (1, 2), f"{input_score.shape}"
    src_ndim = input_score.ndim
    if src_ndim == 1:
        input_score = np.expand_dims(input_score, 1)

    dif_len = (slen - 1) * sitv

    if sitv == 2:
        # CR stream
        offset = dif_len
    else:
        # KR stream
        offset = slen // 2 * sitv

    dst_len = len(input_score) + dif_len

    if dif_len == 0:
        padded_scores = input_score
    elif dif_len > 0:
        padded_scores = np.zeros([dst_len, input_score.shape[1]], dtype=input_score.dtype)
        padded_scores[offset:offset + len(input_score), :] = input_score[:, :]
    else:
        raise NotImplementedError()

    if src_ndim == 1:
        padded_scores = padded_scores.squeeze()
    return padded_scores


def normalize_score(input_score: np.ndarray, ntype: str):
    if ntype == None:
        return input_score

    assert input_score.ndim in (1, 2), f"{input_score.shape}"
    ntype = ntype.lower()

    score: np.ndarray = input_score.copy()
    if score.ndim == 1:
        score = np.expand_dims(score, 1)

    if ntype == 'minmax':
        # MinMax
        denominator = score.max(0, keepdims=True) - score.min(0, keepdims=True)
        assert np.all(denominator != 0)
        score = (score - score.min(0, keepdims=True)) / denominator
    elif ntype == 'meanstd':
        # MeanStd
        denominator = score.std(0, keepdims=True)
        assert np.all(denominator != 0)
        score = (score - score.mean(0, keepdims=True)) / denominator
    elif ntype == 'l2norm':
        # L2Norm
        denominator = np.linalg.norm(score, ord=2, axis=0, keepdims=True)
        assert np.all(denominator != 0)
        score = score / denominator
    else:
        raise NotImplementedError(ntype)

    if input_score.ndim == 1:
        score = score.squeeze()
    assert score.shape == input_score.shape
    return score
