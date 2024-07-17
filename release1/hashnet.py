import torch
from torch import nn
from typing import Tuple


class FeatureTransform(nn.Module):
    def __init__(self, in_dim: Tuple[int]):
        super().__init__()
        self.__in_dim = in_dim
        self.feature_norm = nn.LayerNorm(self.ddim, elementwise_affine=False)

    @property
    def ddim(self) -> int:
        dim = 1
        for _d in self.__in_dim:
            dim *= _d
        return dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, T, H, W, C]

        x = x.flatten(1)  # [batch, ddim]
        x = self.feature_norm(x)
        return x  # [batch, ddim]


class _HashLayer(nn.Module):
    def __init__(self, in_dim: int, len_hash_code: int):
        super().__init__()
        # self.hash_layer = nn.Linear(in_dim, len_hash_code, bias=False)
        # nn.init.xavier_uniform_(self.hash_layer.weight)

        self.hash_layer = nn.Sequential(nn.Linear(in_dim, len_hash_code, bias=False),
                                        nn.LayerNorm(len_hash_code, elementwise_affine=False))
        self.binary_func = nn.Sigmoid()
        nn.init.xavier_uniform_(self.hash_layer[0].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, ddim]
        _hash_code = self.hash_layer(x)
        _hash_code = self.binary_func(_hash_code)

        return _hash_code  # [batch, len_hash_code]


class HashEncoder(nn.Module):
    def __init__(self, in_dim: int, num_hash_layer: int, len_hash_code: int):
        super().__init__()
        self.hash_layers = nn.ModuleList([_HashLayer(in_dim, len_hash_code) for i in range(num_hash_layer)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, ddim]
        hash_codes = torch.stack([layer(x) for layer in self.hash_layers], 1)
        return hash_codes  # [batch, num_hash_layer, len_hash_code]


class HashNet(nn.Module):
    def __init__(self, feat_dim: Tuple[int], num_hash_layer: int, len_hash_code: int):
        super().__init__()
        self.__feat_dim = feat_dim
        self.feature_transform = FeatureTransform(in_dim=feat_dim)
        self.hash_encoder = HashEncoder(self.feature_transform.ddim, num_hash_layer, len_hash_code)

        self.__num_hash_layer = num_hash_layer
        self.__len_hash_code = len_hash_code

        self.__init_check = [True]

    @property
    def num_hash_layer(self) -> int:
        return self.__num_hash_layer

    @property
    def len_hash_code(self) -> int:
        return self.__len_hash_code

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, T, H, W, C]
        if self.__init_check[0]:
            if not list(x.shape[1:]) == self.__feat_dim:
                raise RuntimeError(f"{x.shape}[1:] != {self.__feat_dim}")
            self.__init_check[0] = False

        x = self.feature_transform(x)  # [batch, ddim]
        hash_codes: torch.Tensor = self.hash_encoder(x)  # [batch, num_hash_layer, len_hash_code]

        if self.training:
            long_hash_code = torch.flatten(hash_codes, 1)  # [batch, num_hash_layer * len_hash_code]
            return long_hash_code

        return hash_codes


def mutual_loss(hcode: torch.Tensor):
    """
    Args:
        hcode (torch.Tensor): [batch, b, r]
    """
    # use `torch.tril()` or `torch.triu()`
    b = hcode.shape[1]
    r = hcode.shape[2]
    hcode = hcode / hcode.norm(p=2, dim=-1, keepdim=True)
    sim_mat = torch.matmul(hcode, hcode.transpose(1, 2)).div(r)
    _pos_list = []
    for _row in range(b):
        for _col in range(_row):
            _pos_list.append((_row, _col))
    assert len(_pos_list) == (b**2 - b) / 2
    _pos_d1, _pos_d2 = list(zip(*_pos_list))
    _loss_mat = sim_mat[:, _pos_d1, _pos_d2]
    return _loss_mat.mean()
