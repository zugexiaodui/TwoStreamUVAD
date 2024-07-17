from os import listdir
from os.path import join, exists

import torch
from torch.utils.data.dataset import Dataset as tc_Dataset
from torchvision.io.image import read_image
from torchvision.io.video import read_video
from torchvision import transforms as T
from torchvision.transforms import functional as F

from typing import Tuple, Dict, List
from collections import OrderedDict
import random


class TrainingSetFrame(tc_Dataset):
    def __init__(self, root_dir: str, snippet_len: int, snippet_itv: int, nzfill: int, iterations: int, fixed_3crop: bool):
        '''
        root_dir: the dir containing extracted frames, e.g., 'root_dir'/01_001/[000000.jpg, ...]
        snippet_len: length of a snippet
        snippet_itv: sampling rate (interval between frames)
        nzfill: length of the frame name, e.g., '000000.jpg': nzfill=6
        '''
        super().__init__()
        self.root_dir = root_dir
        self.snippet_len = snippet_len
        self.snippet_itv = snippet_itv
        self.nzfill = nzfill
        self.__fixed_3crop = fixed_3crop

        assert iterations > 0, f"{iterations}"
        self.__iterations = iterations

        self.vid_len_dict = self._get_video_length()
        self.vid_name_list = sorted(list(self.vid_len_dict.keys()))

        self.sta_frm_dict = self._setup_start_frm_idx()

        self._tsfm = self._get_tsfm()

    def _get_video_length(self) -> Dict[str, int]:
        vid_len_dict = {}
        vid_names = listdir(self.root_dir)
        for vid_name in vid_names:
            vid_len_dict[vid_name] = len(listdir(join(self.root_dir, vid_name)))
        return vid_len_dict

    def _setup_start_frm_idx(self) -> OrderedDict:
        vid_name_list = sorted(listdir(self.root_dir))
        vid_sta_frm_dict = OrderedDict()
        for vid_name in vid_name_list:
            vid_sta_frm_dict[vid_name] = self.vid_len_dict[vid_name] - 1 - (self.snippet_len - 1) * self.snippet_itv
        return vid_sta_frm_dict

    def sample_frms_idx(self, sta_frm_idx: int) -> torch.Tensor:
        frm_idx_list = [sta_frm_idx + i * self.snippet_itv for i in range(0, self.snippet_len)]
        return frm_idx_list

    def _load_snippet(self, vid_name: str, idx_t: int):
        idx_t_list = self.sample_frms_idx(idx_t)
        snippet = torch.stack([read_image(join(self.root_dir, vid_name, f"{str(_idx).zfill(self.nzfill)}.jpg")) for _idx in idx_t_list], 0)
        snippet = torch.as_tensor(snippet, dtype=torch.float32) / 255.
        return snippet

    @staticmethod
    def _get_tsfm():
        tsfm = T.Compose([
            T.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225], inplace=False),
            T.Resize(256),
        ])
        return tsfm

    @staticmethod
    def _3crop(snippet: torch.Tensor, idx_s: int):
        height, width = snippet.shape[-2], snippet.shape[-1]
        assert width > height, f"{width}, {height}"
        assert idx_s in range(0, 3), f"{idx_s}"

        x_offset = (width - height) // 2
        if idx_s == 0:
            x_offset = 0
        elif idx_s == 2:
            x_offset = width - height

        snippet = snippet[..., x_offset: x_offset + height]
        return snippet

    @staticmethod
    def _rand_crop(snippet: torch.Tensor, crop_pos: float):
        height, width = snippet.shape[-2], snippet.shape[-1]
        assert width > height, f"{width}, {height}"

        x_offset = round((width - height) * crop_pos)
        snippet = F.crop(snippet, 0, x_offset, height, height)
        return snippet

    def _preprocess(self, snippet: torch.Tensor, crop_pos: float):
        '''
        snippet: [T,C,H,W]
        idx_s: \in {0, 1, 2}
        crop_pos: \in [0, 1)
        => [C,T,H,W]
        '''
        assert 0 <= crop_pos < 1, f"{crop_pos}"

        snippet = self._tsfm(snippet)
        if self.__fixed_3crop:
            snippet = self._3crop(snippet, int(crop_pos * 3))
        else:
            snippet = self._rand_crop(snippet, crop_pos)
        snippet = snippet.permute(1, 0, 2, 3)
        return snippet

    def __getitem__(self, idx) -> torch.Tensor:
        vid_name = self.vid_name_list[idx % len(self.vid_name_list)]

        sta_t = random.randint(0, self.sta_frm_dict[vid_name])
        sta_s = random.random()
        snippet = self._load_snippet(vid_name, sta_t)
        snippet = self._preprocess(snippet, sta_s)

        inputs = snippet[:, :-1, :, :]
        target = torch.unsqueeze(snippet[:, -1, :, :], 1)

        return inputs, target

    def __len__(self):
        return len(self.vid_name_list) * self.__iterations


class TestingSetFrame(TrainingSetFrame):
    def __init__(self, root_dir: str, snippet_len: int, snippet_itv: int, nzfill: int, to_gpu_thres: int = 0, gpu_id: int = 0):
        super().__init__(root_dir, snippet_len, snippet_itv, nzfill, 1, True)
        self.to_gpu_thres = to_gpu_thres
        self.gpu_id = gpu_id

    def _preprocess(self, snippet: torch.Tensor, s_idx):
        '''
        snippet: [T,C,H,W]
        idx_s: \in {0, 1, 2}
        => [C,T,H,W]
        '''

        snippet = self._tsfm(snippet)
        snippet = self._3crop(snippet, s_idx)
        snippet = snippet.permute(1, 0, 2, 3)
        return snippet

    def _sample_all_frms(self, vid_name) -> torch.Tensor:
        '''
        => [T,C,H,W]
        '''
        frm_path_list = [join(self.root_dir, vid_name, f"{str(frm_idx).zfill(self.nzfill)}.jpg") for frm_idx in range(self.vid_len_dict[vid_name])]
        vid_frms = torch.stack([read_image(frm_path) for frm_path in frm_path_list], 0)

        if self.to_gpu_thres < 0:
            vid_frms = vid_frms.cuda(self.gpu_id)
        elif self.to_gpu_thres > 0:
            if len(frm_path_list) <= self.to_gpu_thres:
                vid_frms = vid_frms.cuda(self.gpu_id)

        vid_frms = torch.as_tensor(vid_frms, dtype=torch.float32) / 255.
        return vid_frms

    def __getitem__(self, idx) -> Tuple[str, torch.Tensor]:
        vid_name = self.vid_name_list[idx]
        vid_frms = self._sample_all_frms(vid_name)

        vid_stack = torch.stack([self._preprocess(vid_frms, s_idx) for s_idx in range(3)], 0)  # [3,C,T,H,W]
        return vid_name, vid_stack

    def __len__(self):
        return len(self.vid_name_list)


class TestingSetVideo(TrainingSetFrame):
    def __init__(self, root_dir: str, snippet_len: int, snippet_itv: int, nzfill: int, to_gpu_thres: int = 0, gpu_id: int = 0, vid_dir: str = ''):
        super().__init__(root_dir, snippet_len, snippet_itv, nzfill, 1, True)
        self.to_gpu_thres = to_gpu_thres
        self.gpu_id = gpu_id
        self.vid_dir = vid_dir

    def _preprocess(self, snippet: torch.Tensor, s_idx):
        '''
        snippet: [T,C,H,W]
        idx_s: \in {0, 1, 2}
        => [C,T,H,W]
        '''

        snippet = self._tsfm(snippet)
        snippet = self._3crop(snippet, s_idx)
        snippet = snippet.permute(1, 0, 2, 3)
        return snippet

    def _sample_all_frms(self, vid_name) -> torch.Tensor:
        '''
        => [T,C,H,W]
        '''
        vid_path = join(self.vid_dir, f'{vid_name}.avi')
        if not exists(vid_path):
            raise FileNotFoundError(f"{vid_path}")

        vid_frms = read_video(vid_path)[0]  # [T, H, W, C]
        vid_frms = vid_frms.permute(0, 3, 1, 2)
        vid_frms = F.resize(vid_frms, 256)

        if self.to_gpu_thres < 0:
            vid_frms = vid_frms.cuda(self.gpu_id)
        elif self.to_gpu_thres > 0:
            if len(vid_frms) <= self.to_gpu_thres:
                vid_frms = vid_frms.cuda(self.gpu_id)

        vid_frms = torch.as_tensor(vid_frms, dtype=torch.float32) / 255.
        return vid_frms

    def __getitem__(self, idx) -> Tuple[str, torch.Tensor]:
        vid_name = self.vid_name_list[idx]
        vid_frms = self._sample_all_frms(vid_name)

        vid_stack = torch.stack([self._preprocess(vid_frms, s_idx) for s_idx in range(3)], 0)  # [3,C,T,H,W]
        return vid_name, vid_stack

    def __len__(self):
        return len(self.vid_name_list)


class TrainingSetFeature(tc_Dataset):
    def __init__(self, root_dir: str, t_rand_range: int, iterations: int):
        '''
        root_dir: the dir containing training data (snippet-level-packaged files)
        t_rand_range: a nearby randomly sampling range
        iterations: to simulate how many epochs
        '''
        super().__init__()
        self.root_dir = root_dir

        assert t_rand_range >= 0, f"{t_rand_range}"
        self.rand_t = t_rand_range

        assert iterations > 0, f"{iterations}"
        self.__iterations = iterations

        self.__n_crops = 3

        self.feat_info = self._get_feat_info()

        self.idx2name_map = list(self.feat_info.keys())

    def _get_feat_info(self) -> OrderedDict:
        '''
        Get the information of features, e.g. {'01_001': [1, 2, ..., 732], ...}
        '''
        feat_info = OrderedDict()
        for vid_name in sorted(listdir(self.root_dir)):
            feat_info[vid_name] = list(range(len(listdir(join(self.root_dir, vid_name)))))
        return feat_info

    def _load_snippet(self, vid_name, t_idx, s_idx):
        '''
        Load a snippet according to the given video name and index.
        '''
        pth_path = join(self.root_dir, vid_name, f'{str(t_idx).zfill(6)}.pth')
        return torch.load(pth_path)[s_idx]

    def _sample_temporal_spatial_idx(self, vid_name) -> Tuple[int, int]:
        '''
        Choose a snippet index from a video.
        '''
        t_idx = random.choice(self.feat_info[vid_name])
        s_idx = random.choice(range(self.__n_crops))
        return (t_idx, s_idx)

    def _two_transform(self, _vid_name: str, t_idx: int, s_idx: int) -> torch.Tensor:
        """
        Choose a temporally nearby snippet in the video.
        """
        _t_idx = random.choice(range(max(t_idx - self.rand_t, 0), min(t_idx + self.rand_t, len(self.feat_info[_vid_name]) - 1)))
        _s_idx = random.choices(range(self.__n_crops), k=1)[0]

        return self._load_snippet(_vid_name, _t_idx, _s_idx)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Sample two snippets. One is the `idx`-th snippet and another one is its nearby snippet.
        '''
        vid_idx = idx % len(self.idx2name_map)

        _vid_name = self.idx2name_map[vid_idx]
        _t_idx, _s_idx = self._sample_temporal_spatial_idx(_vid_name)

        _snippet_0: torch.Tensor = self._load_snippet(_vid_name, _t_idx, _s_idx)
        _snippet_1 = self._two_transform(_vid_name, _t_idx, _s_idx)

        return _snippet_0, _snippet_1

    def __len__(self):
        '''
        The length of this dataset is also determined by 'iterations'.
        '''
        return len(self.feat_info) * self.__iterations


class TestingSetFeature(tc_Dataset):
    def __init__(self, root_dir: str):
        '''
        root_dir: the dir containing testing data (video-level-packaged files)
        '''
        super().__init__()
        self.root_dir = root_dir

        self.feat_info = self._get_feat_info()

    def _get_feat_info(self) -> List[str]:
        '''
        Get the video names, e.g., ['01_0014', '01_0015', ...]
        '''
        feat_info = []
        for pth_name in sorted(listdir(self.root_dir)):
            vid_name = pth_name.split('.')[0]
            feat_info.append(vid_name)
        return feat_info

    def __getitem__(self, idx) -> Tuple[str, torch.Tensor]:
        '''
        Return a full video.
        '''
        _vid_name = self.feat_info[idx]
        _snippet: torch.Tensor = torch.load(join(self.root_dir, _vid_name + '.pth'))
        return _vid_name, _snippet

    def __len__(self):
        return len(self.feat_info)
