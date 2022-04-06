# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import torch.nn as nn
import torch

from .builder import SPPE
from .layers.DUC import DUC
from .layers.SE_Resnet import SEResnet
from alfred.dl.torch.common import device


def gather(input, dim, index):
    import itertools
    import operator

    indices = [torch.arange(size, device=index.device) for size in index.shape]
    indices = list(torch.meshgrid(*indices))
    indices[dim] = index
    sizes = list(
        reversed(list(itertools.accumulate(reversed(input.shape), operator.mul)))
    )
    index = sum((index * size for index, size in zip(indices, sizes[1:] + [1])))
    output = input.flatten()[index]
    return output


def get_max_pred_cuda(heatmaps):
    # v, i = torch.max(heatmaps, dim=1)
    # maxvals, ii = torch.max(v, dim=1)
    # iia = ii.unsqueeze(-1)
    # iw = gather(i, 1, iia)
    # print('iia: ', iia.shape)
    # print('iw: ', iw.shape)

    # preds = torch.cat([iia, iw], dim=1)
    # print('preds: ', preds.shape)
    # maxvals = maxvals.unsqueeze(-1)
    # print('maxvals: ', maxvals.shape)

    # mask = maxvals > 0
    # pred_mask = torch.cat([mask, mask], dim=1)
    # preds *= pred_mask
    # print(preds.shape)
    # print(maxvals.shape)

    v, i = torch.max(heatmaps, dim=2)
    maxvals, ii = torch.max(v, dim=2)
    iia = ii.unsqueeze(-1)
    iw = gather(i, 2, iia)

    preds = torch.cat([iia, iw], dim=2)
    # maxvals = maxvals.unsqueeze(-1)
    mask = maxvals > 0
    mask = mask.unsqueeze(-1)
    pred_mask = torch.cat([mask, mask], dim=2)
    preds *= pred_mask
    return preds, maxvals


@SPPE.register_module
class FastPose(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(FastPose, self).__init__()
        self._preset_cfg = cfg["PRESET"]
        if "ONNX_EXPORT" in cfg.keys():
            self.onnx_export = cfg["ONNX_EXPORT"]
        if "CONV_DIM" in cfg.keys():
            self.conv_dim = cfg["CONV_DIM"]
        else:
            self.conv_dim = 128
        if "DCN" in cfg.keys():
            stage_with_dcn = cfg["STAGE_WITH_DCN"]
            dcn = cfg["DCN"]
            self.preact = SEResnet(
                f"resnet{cfg['NUM_LAYERS']}", dcn=dcn, stage_with_dcn=stage_with_dcn
            )
        else:
            self.preact = SEResnet(f"resnet{cfg['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm  # noqa: F401,F403

        assert cfg["NUM_LAYERS"] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")

        model_state = self.preact.state_dict()
        state = {
            k: v
            for k, v in x.state_dict().items()
            if k in self.preact.state_dict()
            and v.size() == self.preact.state_dict()[k].size()
        }
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2, norm_layer=norm_layer)
        if cfg["NUM_LAYERS"] == 18:
            self.duc1 = DUC(128, 1024, upscale_factor=2, norm_layer=norm_layer)
        else:
            self.duc1 = DUC(512, 1024, upscale_factor=2, norm_layer=norm_layer)
        if self.conv_dim == 256:
            self.duc2 = DUC(256, 1024, upscale_factor=2, norm_layer=norm_layer)
        else:
            self.duc2 = DUC(256, 512, upscale_factor=2, norm_layer=norm_layer)
        self.conv_out = nn.Conv2d(
            self.conv_dim,
            self._preset_cfg["NUM_JOINTS"],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # add normalizer
        pixel_mean = torch.as_tensor([0.406, 0.457, 0.480]).view(3, 1, 1).to(device)
        pixel_std = torch.as_tensor([1, 1, 1]).view(3, 1, 1).to(device)
        self.normalizer = lambda x: (x / 255.0 - pixel_mean) / pixel_std

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            # do normalize and transpose here
            # wrap them into model, then inference only need **resize**
            print("normalize and transpose wrapped into onnx.")
            # the input is RGB
            x = x.permute(0, 3, 1, 2)
            x = self.normalizer(x)
            print(x.shape)
            # x = [self.normalizer(xi) for xi in x]

        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        if torch.onnx.is_in_onnx_export():
            print("out shap: ", out.shape)
            print("[WARN] you are in onnx export.")
            coords, maxvals = get_max_pred_cuda(out)
            return coords, maxvals
        else:
            return out

    def _initialize(self):
        for m in self.conv_out.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
