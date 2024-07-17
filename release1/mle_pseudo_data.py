import os
from os import listdir, mkdir, makedirs
from os.path import join, exists, basename, dirname
from typing import List, Tuple, Dict
import random
from collections import OrderedDict
import numpy as np
from math import ceil
from shutil import copyfile
import cv2
from PIL import Image

random.seed(1)
np.random.seed(1)


def select_video_names(vid_dir: str, num: int, rand_seed=0) -> List[str]:
    '''
    选择一部分视频
    '''
    all_vid_name_list = sorted(listdir(vid_dir))

    # Random
    rand_state = random.Random(rand_seed)

    # Scene
    dst_vid_name_list = []
    scene_list_dict = OrderedDict()
    for vid_name in all_vid_name_list:
        scene_name = vid_name.split('_')[0]
        if not scene_name in scene_list_dict:
            scene_list_dict[scene_name] = []
        scene_list_dict[scene_name].append(vid_name)
    for scene_name, vid_name_list in scene_list_dict.items():
        rand_state.shuffle(vid_name_list)
        assert len(vid_name_list) >= num, scene_name
        dst_vid_name_list.extend(vid_name_list[:num])

    return dst_vid_name_list


def replace_frame(frm_idx_array: np.ndarray, anomaly_ratio: float = 0.4):
    num_frm = len(frm_idx_array)
    assert 0 < int(num_frm * anomaly_ratio) < num_frm

    anom_beg_idx = int((num_frm - num_frm * anomaly_ratio) / 2)
    anom_end_idx = int(num_frm - anom_beg_idx)

    frm_idx_array[anom_beg_idx:anom_end_idx] += np.random.randint(-20, 20, size=anom_end_idx - anom_beg_idx, dtype=frm_idx_array.dtype)
    gt_array = np.zeros_like(frm_idx_array)
    gt_array[anom_beg_idx:anom_end_idx] = 1

    return frm_idx_array, gt_array


def move_patch(frm_idx_array: np.ndarray, src_vid_dir: str, dst_vid_dir: str, anomaly_ratio: float = 0.5):
    num_frm = len(frm_idx_array)
    assert 0 < int(num_frm * anomaly_ratio) < num_frm

    anom_beg_idx = int((num_frm - num_frm * anomaly_ratio) / 2)
    anom_end_idx = int(num_frm - anom_beg_idx)

    anom_array = frm_idx_array[anom_beg_idx:anom_end_idx]

    for frm_idx in frm_idx_array:
        src_frm_path = join(src_vid_dir, f"{str(frm_idx).zfill(4)}.jpg")
        dst_frm_path = join(dst_vid_dir, f"{str(frm_idx).zfill(4)}.jpg")
        assert exists(src_frm_path), src_frm_path
        # if exists(dst_frm_path):
        #     continue
        makedirs(dirname(dst_frm_path), exist_ok=True)

        src_img = cv2.imread(src_frm_path)
        src_img = cv2.flip(src_img, 1)
        src_img = Image.fromarray(src_img, mode='RGB')
        _ang = random.randint(2, 5)
        src_img = src_img.rotate(_ang, Image.BICUBIC)
        src_img = np.array(src_img)

        if frm_idx in anom_array:
            # anom_img = _move_patch(src_img)
            fut_img1 = cv2.imread(join(src_vid_dir, f"{str(frm_idx+2).zfill(4)}.jpg"))
            fut_img1 = cv2.flip(fut_img1, 1)
            fut_img1 = Image.fromarray(fut_img1, mode='RGB')
            fut_img1 = fut_img1.rotate(_ang, Image.BICUBIC)
            fut_img1 = np.array(fut_img1)
            anom_img = np.asarray((np.asarray(src_img, np.float64) +
                                   np.asarray(fut_img1, np.float64)
                                   ) / 2, np.uint8)

            cv2.imwrite(dst_frm_path, anom_img, (cv2.IMWRITE_JPEG_QUALITY, 100))
        else:
            cv2.imwrite(dst_frm_path, src_img, (cv2.IMWRITE_JPEG_QUALITY, 100))

    gt_array = np.zeros_like(frm_idx_array)
    gt_array[anom_beg_idx:anom_end_idx] = 1

    return gt_array


def compose_gt_files(gt_root: str, debug=False):
    if debug:
        return
    gt_dict = {}
    for fname in sorted(listdir(gt_root)):
        gt_dict[fname.split('.')[0]] = np.load(join(gt_root, fname))
    np.savez(join(dirname(grt_root), 'gt.npz'), **gt_dict)


def generate_pseudo_1(src_vid_dir: str, dst_vid_dir: str, grt_root: str, mode: str, seg_param, debug=False):
    '''
    对每个视频进行伪异常处理，保存视频并返回gt标签
    处理方式为，把一些帧替换为附近的几帧，标记为异常
    '''
    if not debug:
        if exists(dst_vid_dir):
            for fname in listdir(dst_vid_dir):
                os.remove(join(dst_vid_dir, fname))

    src_frm_list = sorted(listdir(src_vid_dir))
    num_frm = len(src_frm_list)
    src_frm_idx_array = np.arange(0, num_frm)

    print(basename(src_vid_dir), num_frm)

    if mode == 'fixed_num':
        num_segment = seg_param
    elif mode == 'fixed_len':
        each_segment_len = seg_param
        num_segment = ceil(num_frm / each_segment_len)

    segment_list = np.array_split(src_frm_idx_array, num_segment)

    dst_frm_idx_array = []
    dst_gt_array = []
    for segment_i in segment_list:
        anom_segment, anom_gt = replace_frame(segment_i)
        dst_frm_idx_array.append(anom_segment)
        dst_gt_array.append(anom_gt)
    dst_frm_idx_array = np.concatenate(dst_frm_idx_array)
    dst_gt_array = np.concatenate(dst_gt_array)

    assert len(dst_frm_idx_array) == len(dst_gt_array) == len(src_frm_idx_array)

    for src_frm_idx, dst_frm_idx in enumerate(dst_frm_idx_array):
        src_frm_path = join(src_vid_dir, src_frm_list[dst_frm_idx])
        dst_frm_path = join(dst_vid_dir, src_frm_list[src_frm_idx])

        if debug:
            print(f"{src_frm_path} ==> {dst_frm_path}")
        else:
            makedirs(dirname(dst_frm_path), exist_ok=True)
            copyfile(src_frm_path, dst_frm_path)

    gt_path = join(grt_root, f"{basename(dst_vid_dir)}.npy")

    if debug:
        # print(gt_path)
        pass
    else:
        makedirs(dirname(gt_path), exist_ok=True)
        np.save(gt_path, dst_gt_array)


def generate_pseudo_videos(src_vid_dir: str, dst_vid_dir: str, grt_root: str, mode: str, seg_param, debug=False):
    '''
    对每个视频进行伪异常处理，保存视频并返回gt标签
    处理方式为，把一个patch复制位移一下作为异常
    '''
    if not debug:
        ans = input("是否删除已存在文件？")
        if ans == 'y':
            if exists(dst_vid_dir):
                for fname in listdir(dst_vid_dir):
                    os.remove(join(dst_vid_dir, fname))
                print(f"{dst_vid_dir} 删除完成")

    num_frm = len(listdir(src_vid_dir))
    src_frm_idx_array = np.arange(0, num_frm)

    print(basename(src_vid_dir), num_frm)

    if mode == 'fixed_num':
        num_segment = seg_param
    elif mode == 'fixed_len':
        each_segment_len = seg_param
        num_segment = ceil(num_frm / each_segment_len)

    segment_list = np.array_split(src_frm_idx_array, num_segment)

    dst_gt_array = []
    for segment_i in segment_list:
        anom_gt = move_patch(segment_i, src_vid_dir, dst_vid_dir)
        dst_gt_array.append(anom_gt)
    dst_gt_array = np.concatenate(dst_gt_array)

    assert len(dst_gt_array) == len(src_frm_idx_array)

    gt_path = join(grt_root, f"{basename(dst_vid_dir)}.npy")

    if debug:
        # print(gt_path)
        pass
    else:
        makedirs(dirname(gt_path), exist_ok=True)
        np.save(gt_path, dst_gt_array)


if __name__ == "__main__":
    anomaly_type = 'forST'

    src_root = "../data.ST/Train"
    dst_root = f"../data.ST/for_mle/{anomaly_type}/Train"
    grt_root = f"../data.ST/for_mle/{anomaly_type}/gts"

    assert anomaly_type

    debug = False
    if exists(dst_root):
        if input(f"已存在：{dst_root}，要继续吗？") != 'y':
            exit()

    vid_name_list = select_video_names(src_root, num=5)

    for vid_name in vid_name_list:
        generate_pseudo_videos(join(src_root, vid_name), join(dst_root, vid_name), grt_root, 'fixed_num', 1, debug)
    compose_gt_files(grt_root, debug)
