import torch
from torch import Tensor
from torch import nn
from torch.nn import init


class _RebuildResNetBasicHead(nn.Module):
    def __init__(self, dim_in, pool_size):
        super(_RebuildResNetBasicHead, self).__init__()
        assert (len({len(pool_size), len(dim_in)}) == 1), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

    def forward(self, inputs):
        assert (len(inputs) == self.num_pathways), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        # x = x.permute((0, 2, 3, 4, 1))
        return x


def rebuild_network(model: nn.Module, encoder_rmrelu: bool):
    if encoder_rmrelu:
        model.s5.pathway0_res2.relu = nn.Identity()
    model.head = _RebuildResNetBasicHead([2048], [[4, 7, 7]])
    return model


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # [1, 2048, 1, 2, 2]
        self.up1 = nn.Sequential(
            nn.Conv3d(2048, 512, 1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(512, 512, (1, 2, 2), (1, 2, 2)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),

            nn.Conv3d(512, 512, 1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(512, 2048, (1, 2, 2), (1, 2, 2)),
            nn.BatchNorm3d(2048),
            nn.LeakyReLU(),
        )  # [1, 2048, 1, 8, 8]
        self.uconv1 = nn.Sequential(
            nn.Conv3d(2048 + 2048, 512, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),
            nn.Conv3d(512, 512, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),
            nn.Conv3d(512, 2048, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(2048),
            nn.LeakyReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(2048, 1024, (1, 2, 2), (1, 2, 2)),
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(),
        )  # [1, 1024, 1, 16, 16]
        self.uconv2 = nn.Sequential(
            nn.Conv3d(1024 + 1024, 256, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            nn.Conv3d(256, 256, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            nn.Conv3d(256, 1024, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(1024, 512, (1, 2, 2), (1, 2, 2)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),
        )  # [1, 512, 1, 32, 32]
        self.uconv3 = nn.Sequential(
            nn.Conv3d(512 + 512, 128, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Conv3d(128, 512, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, (1, 2, 2), (1, 2, 2)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
        )  # [1, 256, 1, 64, 64]
        self.uconv4 = nn.Sequential(
            nn.Conv3d(256 + 256, 64, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 256, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, (1, 2, 2), (1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
        )  # [1, 128, 1, 128, 128]
        self.uconv5 = nn.Sequential(
            nn.Conv3d(128, 32, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(32, 32, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(32, 128, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, (1, 2, 2), (1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
        )  # [1, 64, 1, 256, 256]

        self.uconv6 = nn.Sequential(
            nn.Conv3d(64, 64, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 3, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(),
            nn.Conv3d(3, 3, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
        )

        self.init_params()

    def init_params(self, init_type='kaiming'):
        init_gain = 1.0

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data)
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x: torch.Tensor):
        x = self.up1(x)
        x = self.uconv1(x)

        x = self.up2(x)
        x = self.uconv2(x)

        x = self.up3(x)
        x = self.uconv3(x)

        x = self.up4(x)
        x = self.uconv4(x)

        x = self.up5(x)
        x = self.uconv5(x)

        x = self.up6(x)
        x = self.uconv6(x)

        return x


class STUNet(nn.Module):
    def __init__(self, encoder_path: str, encoder_reinit: bool, encoder_rmrelu: bool, only_feat: bool = False):
        super().__init__()
        self.encoder = rebuild_network(torch.load(encoder_path, map_location=f'cpu'), encoder_rmrelu)
        self.only_feat = only_feat

        if encoder_reinit:
            self.init_params_encoder()

        self.decoder = Decoder()

        self.tconv3 = nn.Conv3d(256, 256, (4, 1, 1))
        self.tconv4 = nn.Conv3d(512, 512, (4, 1, 1))
        self.tconv5 = nn.Conv3d(1024, 1024, (4, 1, 1))
        self.tconv6 = nn.Conv3d(2048, 2048, (4, 1, 1))

    def init_params_encoder(self, init_type='kaiming'):
        init_gain = 1.0

        def init_func(m):  # define the initialization function
            classname: str = m.__class__.__name__
            if hasattr(m, 'weight'):
                if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                    if init_type == 'normal':
                        init.normal_(m.weight.data, 0.0, init_gain)
                    elif init_type == 'xavier':
                        init.xavier_normal_(m.weight.data, gain=init_gain)
                    elif init_type == 'kaiming':
                        init.kaiming_normal_(m.weight.data)
                    elif init_type == 'orthogonal':
                        init.orthogonal_(m.weight.data, gain=init_gain)
                    else:
                        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                elif classname.find('BatchNorm') != -1:
                    init.constant_(m.weight.data, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.encoder.apply(init_func)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = [x]

        y1: Tensor = self.encoder.s1(x)[0]  # [1, 64, 8, 64, 64]

        y2: Tensor = self.encoder.s2([y1])[0]  # [1, 256, 8, 64, 64]

        y3: Tensor = self.encoder.pathway0_pool(y2)  # [1, 256, 4, 64, 64]

        y4: Tensor = self.encoder.s3([y3])[0]  # [1, 512, 4, 32, 32]

        y5: Tensor = self.encoder.s4([y4])[0]  # [1, 1024, 4, 16, 16]

        y6: Tensor = self.encoder.s5([y5])[0]  # [1, 2048, 4, 8, 8]

        y7: Tensor = self.encoder.head([y6])  # [1, 2048, 1, 2, 2]

        if self.only_feat:
            return y7

        z7 = self.decoder.up1(y7)  # [1, 2048, 1, 8, 8]
        t7 = self.tconv6(y6)
        o7 = self.decoder.uconv1(torch.cat([z7, t7], 1))

        z6 = self.decoder.up2(o7)  # [1, 1024, 1, 16, 16]
        t6 = self.tconv5(y5)
        o6 = self.decoder.uconv2(torch.cat([z6, t6], 1))

        z5 = self.decoder.up3(o6)  # [1, 512, 1, 32, 32]
        t5 = self.tconv4(y4)
        o5 = self.decoder.uconv3(torch.cat([z5, t5], 1))

        z4 = self.decoder.up4(o5)  # [1, 256, 1, 64, 64]
        t4 = self.tconv3(y3)
        o4 = self.decoder.uconv4(torch.cat([z4, t4], 1))

        out = self.decoder.up5(o4)  # [1, 128, 1, 128, 128]
        out = self.decoder.uconv5(out)

        out = self.decoder.up6(out)  # [1, 64, 1, 256, 256]
        out = self.decoder.uconv6(out)

        return out
