import torch
import numpy as np

from collections import defaultdict
from time import time as ttime

from hashnet import HashNet


class iL2SH():
    def __init__(self, hash_net: HashNet, verbose=False):
        self.hash_net = hash_net.cuda()

        self.hash_table = [defaultdict(list) for i in range(hash_net.num_hash_layer)]

        self.__init_check = [True] * 2
        self.__verbose = verbose

    def cvt_key(self, hash_code: np.ndarray) -> str:
        """
        convert a hash_code (e.g. [0 1 0 1]) to string ('0101'). More effecient than `np.array2string()`.
        """
        if self.__init_check[0]:
            if not hash_code.dtype == np.int:
                raise TypeError(f"The dtype of input array must be 'np.int'. {hash_code.dtype}")
            if not len(hash_code.shape) == 1:
                raise TypeError(f"The shape of input array must be '(N,)'. {hash_code.shape}")
            if not (np.all(np.unique(hash_code) == np.array([0, 1], dtype=np.int))
                    or np.all(np.unique(hash_code) == np.array([0], dtype=np.int))
                    or np.all(np.unique(hash_code) == np.array([1], dtype=np.int))):
                raise ValueError(f"It should be only 0 and 1 in the array. {np.unique(hash_code)}")
            self.__init_check[0] = False
        return ''.join(['0' if s == 0 else '1' for s in hash_code])

    def batch_construction(self, vecs: torch.Tensor) -> None:
        """
        Construction. A batch of 1-D vectors.
        vecs: (N, ddim)
        """
        assert isinstance(vecs, torch.Tensor), f"type(vec): '{type(vecs)}'. It should be 'torch.Tensor'"
        # assert len(vecs.shape) == 2

        t0 = ttime()
        if self.__verbose:
            print("Batch hashing ...", end='', flush=True)

        vecs = vecs.cuda()

        with torch.no_grad():
            _hash_codes: torch.Tensor = self.hash_net(vecs)  # [batch, num_hash_layer, len_hash_code], i.e. [N, b, r]

            if self.__init_check[0]:
                assert len(_hash_codes.shape) == 3, _hash_codes.shape
                assert torch.all(_hash_codes >= 0) and torch.all(_hash_codes <= 1), "Please use `sigmoid` function for binaryzation."

        _hash_codes.transpose_(0, 1)  # [b, N, r]
        _hash_codes = _hash_codes.cpu()

        _bin_codes: np.ndarray = np.asarray(torch.round(_hash_codes).numpy(), dtype=np.int)  # [b, N, r]

        t1 = ttime()
        if self.__verbose:
            print(f"{t1-t0:.3f}s", end=' ', flush=True)

        if self.__verbose:
            print("Indexing ...", end=' ', flush=True)

        for _b, (_table_b, _bin_code_b, _real_code_b) in enumerate(zip(self.hash_table, _bin_codes, _hash_codes)):  # b * [N, r]
            for _bin_code_b_n, _real_code_b_n in zip(_bin_code_b, _real_code_b):  # N * [r,]
                _real_code_b_n: torch.Tensor
                _bucket_key = self.cvt_key(_bin_code_b_n)

                if not _bucket_key in _table_b:
                    _table_b[_bucket_key] = [1, _real_code_b_n.clone()]
                    # _table_b[_bucket_key] = [1, _real_code_b_n]
                else:
                    _num, _center = _table_b[_bucket_key]
                    _table_b[_bucket_key][0] = _num + 1
                    _table_b[_bucket_key][1] = (_num / (_num + 1) * _center + _real_code_b_n / (_num + 1)).clone()

        if self.__verbose:
            print(f"{ttime()-t1:.3f}s")

    def batch_retrieval(self, vecs: torch.Tensor) -> torch.Tensor:
        """
        Retrieve. A batch of 1-D vectors.
        vecs: (M, ddim)
        Return (M, num_hash_layer): distance between each vector and query results of every hash layer.
        """
        if self.__init_check[1]:
            assert isinstance(vecs, torch.Tensor), f"'vec({type(vecs)})' should be torch.Tensor"
            self.hash_net = self.hash_net.cuda()

        vecs = vecs.cuda()

        with torch.no_grad():
            _hash_codes: torch.Tensor = self.hash_net(vecs)  # [batch, num_hash_layer, len_hash_code], i.e. [M, b, r]

            if self.__init_check[1]:
                assert len(_hash_codes.shape) == 3, _hash_codes.shape
                assert torch.all(_hash_codes >= 0) and torch.all(_hash_codes <= 1), "Please use `sigmoid` function for binaryzation."
                self.__init_check[1] = False

        _hash_codes = _hash_codes.cpu()

        _bin_codes: np.ndarray = torch.round(_hash_codes).numpy()  # [M, b, r]

        candist_all = []
        for _bin_code_m, _real_code_m in zip(_bin_codes, _hash_codes):  # M * [b, r]
            _candist_b = []
            for _b, (_table_b, _bin_code_m_b, _real_code_m_b) in enumerate(zip(self.hash_table, _bin_code_m, _real_code_m)):  # b * [r,]
                _bucket_val = _table_b[self.cvt_key(_bin_code_m_b)]

                if not _bucket_val == []:
                    _cands_list = _bucket_val[1]
                    _dists: torch.Tensor = torch.norm(_cands_list - _real_code_m_b, p=2, dim=-1)

                    _candist_b.append(_dists.mean())
                else:
                    _candist_b.append(torch.sqrt(torch.as_tensor(self.hash_net.len_hash_code, dtype=torch.float32)))
            _candist_b = torch.as_tensor(_candist_b)  # [b,]

            candist_all.append(_candist_b)

        candist_all = torch.stack(candist_all)  # [M, b]
        return candist_all

    def show_tables(self, num: int = 5):
        '''
        Display the `num` buckets with the most and least numbers of hash codes.
        It can be used after `cvt_list2tensor()`.
        '''
        num_code_dict = {}
        for _b, _table_b in enumerate(self.hash_table):
            num_code_dict = {}
            for _k in _table_b.keys():
                num_code_dict[_k] = len(_table_b[_k])
            sort_vals = sorted(list(num_code_dict.items()), key=lambda x: x[1], reverse=True)
            print(f"table {_b}: {len(sort_vals)} buckets" + '-' * 40)
            most_vals = sort_vals[:num]
            least_vals = sort_vals[-num:]
            for _mk, _mv in most_vals:
                print(f"{_mk}: {_mv}")
            print('......')
            for _mk, _mv in least_vals:
                print(f"{_mk}: {_mv}")
        print('-' * 60)
