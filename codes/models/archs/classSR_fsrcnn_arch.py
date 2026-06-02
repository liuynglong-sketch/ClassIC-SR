import torch
import torch.nn as nn
import torch.nn.functional as F

import models.archs.arch_util as arch_util
from models.archs.FSRCNN_arch import FSRCNN_net


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lastOut = nn.Linear(32, 3)
        self.CondNet = nn.Sequential(
            nn.Conv2d(3, 128, 4, 4),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 32, 1),
        )
        arch_util.initialize_weights([self.CondNet], 0.1)

    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AvgPool2d(out.size()[2])(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        return out


class TinyCNNRouter(nn.Module):
    """Tiny learnable router for on-chip easy/medium/hard branch selection."""

    def __init__(self, in_ch=3, width=16, mid_ch=8, num_classes=3, activation="relu", use_mid_projection=True):
        super(TinyCNNRouter, self).__init__()
        if activation == "prelu":
            act1 = nn.PReLU(width)
            act2 = nn.PReLU(width)
        elif activation == "relu":
            act1 = nn.ReLU(inplace=True)
            act2 = nn.ReLU(inplace=True)
        else:
            raise ValueError("Unsupported TinyCNNRouter activation: {}".format(activation))

        layers = [
            nn.Conv2d(in_ch, width, kernel_size=4, stride=4, padding=0),
            act1,
            nn.Conv2d(width, width, kernel_size=1, stride=1, padding=0),
            act2,
        ]
        out_ch = width
        if use_mid_projection:
            layers.append(nn.Conv2d(width, mid_ch, kernel_size=1, stride=1, padding=0))
            out_ch = mid_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lastOut = nn.Linear(out_ch, num_classes)
        arch_util.initialize_weights([self.features], 0.1)

    def forward(self, x):
        out = self.features(x)
        out = self.pool(out).view(out.size(0), -1)
        return self.lastOut(out)


class SimpleInterpCompNet(nn.Module):
    """Net1: x4 bilinear -> 1x1 -> 3x3."""

    def __init__(self, in_nc=3, out_nc=3, upscale=4, hidden=8):
        super(SimpleInterpCompNet, self).__init__()
        self.tail = nn.Sequential(
            nn.Upsample(scale_factor=upscale, mode="bilinear", align_corners=False),
            nn.Conv2d(in_nc, hidden, kernel_size=1, stride=1, padding=0),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, out_nc, kernel_size=3, stride=1, padding=1),
        )
        arch_util.initialize_weights([self.tail], 0.1)

    def forward(self, x):
        return self.tail(x)


class FSRCNNInterp2xComp2xNet(nn.Module):
    """FSRCNN backbone + x2 bilinear -> 3x3 -> x2 bilinear -> 1x1."""

    def __init__(self, input_channels, upscale, d=64, s=12, m=4):
        super(FSRCNNInterp2xComp2xNet, self).__init__()
        if upscale != 4:
            raise ValueError("FSRCNNInterp2xComp2xNet currently expects upscale=4.")

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )

        layers = [
            nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0), nn.PReLU())
        ]
        for _ in range(m):
            layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        layers.append(nn.PReLU())
        layers.append(
            nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0), nn.PReLU())
        )
        self.body_conv = nn.Sequential(*layers)

        self.tail_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1),
            nn.PReLU(d),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d, input_channels, kernel_size=1, stride=1, padding=0),
        )

        arch_util.initialize_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.body_conv(fea)
        out = self.tail_conv(fea)
        return out


class FSRCNNInterp2xPWComp2xNet(nn.Module):
    """FSRCNN backbone + x2 bilinear -> 1x1 -> 3x3 -> x2 bilinear -> 1x1."""

    def __init__(self, input_channels, upscale, d=64, s=12, m=4):
        super(FSRCNNInterp2xPWComp2xNet, self).__init__()
        if upscale != 4:
            raise ValueError("FSRCNNInterp2xPWComp2xNet currently expects upscale=4.")

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )

        layers = [
            nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0), nn.PReLU())
        ]
        for _ in range(m):
            layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        layers.append(nn.PReLU())
        layers.append(
            nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0), nn.PReLU())
        )
        self.body_conv = nn.Sequential(*layers)

        self.tail_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d, d, kernel_size=1, stride=1, padding=0),
            nn.PReLU(d),
            nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1),
            nn.PReLU(d),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d, input_channels, kernel_size=1, stride=1, padding=0),
        )

        arch_util.initialize_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.body_conv(fea)
        out = self.tail_conv(fea)
        return out


class FSRCNNInterp4xConv3xNet(nn.Module):
    """FSRCNN medium body + x4 bilinear feature interpolation + HR 3x3 RGB projection."""

    def __init__(self, input_channels, upscale, d=36, s=12, m=4):
        super(FSRCNNInterp4xConv3xNet, self).__init__()
        if upscale != 4:
            raise ValueError("FSRCNNInterp4xConv3xNet currently expects upscale=4.")

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )

        layers = [
            nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0), nn.PReLU())
        ]
        for _ in range(m):
            layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        layers.append(nn.PReLU())
        layers.append(
            nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0), nn.PReLU())
        )
        self.body_conv = nn.Sequential(*layers)

        self.tail_conv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(d, input_channels, kernel_size=3, stride=1, padding=1),
        )

        arch_util.initialize_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.body_conv(fea)
        return self.tail_conv(fea)


class FSRCNNInterp2x3x3Interp2x3x3Net(nn.Module):
    """FSRCNN medium body + x2 bilinear -> 3x3 36->36 -> x2 bilinear -> HR 3x3 36->3."""

    def __init__(self, input_channels, upscale, d=36, s=12, m=4):
        super(FSRCNNInterp2x3x3Interp2x3x3Net, self).__init__()
        if upscale != 4:
            raise ValueError("FSRCNNInterp2x3x3Interp2x3x3Net currently expects upscale=4.")

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )

        layers = [
            nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0), nn.PReLU())
        ]
        for _ in range(m):
            layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        layers.append(nn.PReLU())
        layers.append(
            nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0), nn.PReLU())
        )
        self.body_conv = nn.Sequential(*layers)

        self.tail_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1),
            nn.PReLU(d),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d, input_channels, kernel_size=3, stride=1, padding=1),
        )

        arch_util.initialize_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.body_conv(fea)
        return self.tail_conv(fea)


class FSRCNNInterp2x3x3C24Interp2x3x3Net(nn.Module):
    """FSRCNN medium body + x2 -> 3x3 36->24 -> x2 -> HR 3x3 24->3."""

    def __init__(self, input_channels, upscale, d=36, s=12, m=4, mid_channels=24):
        super(FSRCNNInterp2x3x3C24Interp2x3x3Net, self).__init__()
        if upscale != 4:
            raise ValueError("FSRCNNInterp2x3x3C24Interp2x3x3Net currently expects upscale=4.")

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )

        layers = [
            nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0), nn.PReLU())
        ]
        for _ in range(m):
            layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        layers.append(nn.PReLU())
        layers.append(
            nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0), nn.PReLU())
        )
        self.body_conv = nn.Sequential(*layers)

        self.tail_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(mid_channels),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid_channels, input_channels, kernel_size=3, stride=1, padding=1),
        )

        arch_util.initialize_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.body_conv(fea)
        return self.tail_conv(fea)


class FSRCNNInterp2x3x3Interp2x1x1Net(nn.Module):
    """FSRCNN medium body + x2 -> 3x3 36->36 -> x2 -> HR 1x1 36->3."""

    def __init__(self, input_channels, upscale, d=36, s=12, m=4):
        super(FSRCNNInterp2x3x3Interp2x1x1Net, self).__init__()
        if upscale != 4:
            raise ValueError("FSRCNNInterp2x3x3Interp2x1x1Net currently expects upscale=4.")

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )

        layers = [
            nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0), nn.PReLU())
        ]
        for _ in range(m):
            layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        layers.append(nn.PReLU())
        layers.append(
            nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0), nn.PReLU())
        )
        self.body_conv = nn.Sequential(*layers)

        self.tail_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1),
            nn.PReLU(d),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d, input_channels, kernel_size=1, stride=1, padding=0),
        )

        arch_util.initialize_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.body_conv(fea)
        return self.tail_conv(fea)


class FSRCNNBottleneckMR2xNet(nn.Module):
    """FSRCNN medium body + MR bottleneck compensation + final x2 interpolation."""

    def __init__(self, input_channels, upscale, d=36, s=12, m=4, bottleneck=16):
        super(FSRCNNBottleneckMR2xNet, self).__init__()
        if upscale != 4:
            raise ValueError("FSRCNNBottleneckMR2xNet currently expects upscale=4.")

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )

        layers = [
            nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0), nn.PReLU())
        ]
        for _ in range(m):
            layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        layers.append(nn.PReLU())
        layers.append(
            nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0), nn.PReLU())
        )
        self.body_conv = nn.Sequential(*layers)

        self.tail_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d, bottleneck, kernel_size=1, stride=1, padding=0),
            nn.PReLU(bottleneck),
            nn.Conv2d(bottleneck, bottleneck, kernel_size=3, stride=1, padding=1),
            nn.PReLU(bottleneck),
            nn.Conv2d(bottleneck, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        arch_util.initialize_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.body_conv(fea)
        return self.tail_conv(fea)


class SharedLRTrunk(nn.Module):
    """Shared FSRCNN-style LR trunk without the final upsampling tail."""

    def __init__(self, input_channels=3, d=36, s=12, m=4):
        super(SharedLRTrunk, self).__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )

        layers = [
            nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0), nn.PReLU())
        ]
        for _ in range(m):
            layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        layers.append(nn.PReLU())
        layers.append(
            nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0), nn.PReLU())
        )
        self.body_conv = nn.Sequential(*layers)

        arch_util.initialize_weights([self.head_conv, self.body_conv], 0.1)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.body_conv(fea)
        return fea


class EasyInterpHead(nn.Module):
    """x4 interpolation baseline plus tiny residual correction."""

    def __init__(self, in_nc=3, out_nc=3, hidden=8):
        super(EasyInterpHead, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_nc, hidden, kernel_size=1, stride=1, padding=0),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, out_nc, kernel_size=3, stride=1, padding=1),
        )
        arch_util.initialize_weights([self.residual], 0.1)

    def forward(self, x, feat_lr=None):
        hr0 = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        delta = self.residual(hr0)
        return hr0 + delta


class EasyInterpHeadLRHint(nn.Module):
    """Keep the x4 interpolation design, but add a tiny LR pre-correction."""

    def __init__(self, in_nc=3, out_nc=3, hidden=8, hint_hidden=8):
        super(EasyInterpHeadLRHint, self).__init__()
        self.lr_hint = nn.Sequential(
            nn.Conv2d(in_nc, hint_hidden, kernel_size=3, stride=1, padding=1),
            nn.PReLU(hint_hidden),
            nn.Conv2d(hint_hidden, in_nc, kernel_size=1, stride=1, padding=0),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(in_nc, hidden, kernel_size=1, stride=1, padding=0),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, out_nc, kernel_size=3, stride=1, padding=1),
        )
        arch_util.initialize_weights([self.lr_hint, self.residual], 0.1)

    def forward(self, x, feat_lr=None):
        x_refined = x + self.lr_hint(x)
        hr0 = F.interpolate(x_refined, scale_factor=4, mode="bilinear", align_corners=False)
        delta = self.residual(hr0)
        return hr0 + delta


class MediumMRHead(nn.Module):
    """MR-domain residual correction followed by light HR refine."""

    def __init__(self, in_nc=3, d=36):
        super(MediumMRHead, self).__init__()
        self.mr_residual = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1),
            nn.PReLU(d),
            nn.Conv2d(d, in_nc, kernel_size=1, stride=1, padding=0),
        )
        self.hr_refine = nn.Conv2d(in_nc, in_nc, kernel_size=1, stride=1, padding=0)
        arch_util.initialize_weights([self.mr_residual, self.hr_refine], 0.1)

    def forward(self, x, feat_lr):
        mr_base = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        feat_mr = F.interpolate(feat_lr, scale_factor=2, mode="bilinear", align_corners=False)
        delta_mr = self.mr_residual(feat_mr)
        mr = mr_base + delta_mr
        hr0 = F.interpolate(mr, scale_factor=2, mode="bilinear", align_corners=False)
        return hr0 + self.hr_refine(hr0)


class HardMRHead(nn.Module):
    """Stronger MR correction while keeping HR refine extremely light."""

    def __init__(self, in_nc=3, d=36):
        super(HardMRHead, self).__init__()
        self.mr_residual = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=1, stride=1, padding=0),
            nn.PReLU(d),
            nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1),
            nn.PReLU(d),
            nn.Conv2d(d, in_nc, kernel_size=1, stride=1, padding=0),
        )
        self.mr_refine = nn.Conv2d(in_nc, in_nc, kernel_size=3, stride=1, padding=1)
        self.hr_refine = nn.Conv2d(in_nc, in_nc, kernel_size=1, stride=1, padding=0)
        arch_util.initialize_weights([self.mr_residual, self.mr_refine, self.hr_refine], 0.1)

    def forward(self, x, feat_lr):
        mr_base = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        feat_mr = F.interpolate(feat_lr, scale_factor=2, mode="bilinear", align_corners=False)
        delta_mr = self.mr_residual(feat_mr)
        mr = mr_base + delta_mr
        mr = mr + self.mr_refine(mr)
        hr0 = F.interpolate(mr, scale_factor=2, mode="bilinear", align_corners=False)
        return hr0 + self.hr_refine(hr0)


class HardMRHeadV2(nn.Module):
    """Two-stage MR residual correction with light HR refine."""

    def __init__(self, in_nc=3, d=36, stage_width=48, mr_feat=16):
        super(HardMRHeadV2, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(d, stage_width, kernel_size=1, stride=1, padding=0),
            nn.PReLU(stage_width),
            nn.Conv2d(stage_width, stage_width, kernel_size=3, stride=1, padding=1),
            nn.PReLU(stage_width),
            nn.Conv2d(stage_width, in_nc, kernel_size=1, stride=1, padding=0),
        )
        self.mr_encode = nn.Sequential(
            nn.Conv2d(in_nc, mr_feat, kernel_size=3, stride=1, padding=1),
            nn.PReLU(mr_feat),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(d + mr_feat, stage_width, kernel_size=1, stride=1, padding=0),
            nn.PReLU(stage_width),
            nn.Conv2d(stage_width, stage_width, kernel_size=3, stride=1, padding=1),
            nn.PReLU(stage_width),
            nn.Conv2d(stage_width, in_nc, kernel_size=1, stride=1, padding=0),
        )
        self.mr_refine = nn.Conv2d(in_nc, in_nc, kernel_size=3, stride=1, padding=1)
        self.hr_refine = nn.Conv2d(in_nc, in_nc, kernel_size=1, stride=1, padding=0)
        arch_util.initialize_weights(
            [self.stage1, self.mr_encode, self.stage2, self.mr_refine, self.hr_refine], 0.1
        )

    def forward(self, x, feat_lr):
        mr_base = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        feat_mr = F.interpolate(feat_lr, scale_factor=2, mode="bilinear", align_corners=False)

        delta1 = self.stage1(feat_mr)
        mr1 = mr_base + delta1

        mr1_feat = self.mr_encode(mr1)
        feat_stage2 = torch.cat([feat_mr, mr1_feat], dim=1)
        delta2 = self.stage2(feat_stage2)
        mr2 = mr1 + delta2

        mr2 = mr2 + self.mr_refine(mr2)
        hr0 = F.interpolate(mr2, scale_factor=2, mode="bilinear", align_corners=False)
        return hr0 + self.hr_refine(hr0)


class HardMRHeadV3Lite(nn.Module):
    """Cheaper two-stage MR correction with image-driven stage2."""

    def __init__(self, in_nc=3, d=36, stage1_width=40, stage2_width=8):
        super(HardMRHeadV3Lite, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(d, stage1_width, kernel_size=1, stride=1, padding=0),
            nn.PReLU(stage1_width),
            nn.Conv2d(stage1_width, stage1_width, kernel_size=3, stride=1, padding=1),
            nn.PReLU(stage1_width),
            nn.Conv2d(stage1_width, in_nc, kernel_size=1, stride=1, padding=0),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(in_nc, stage2_width, kernel_size=3, stride=1, padding=1),
            nn.PReLU(stage2_width),
            nn.Conv2d(stage2_width, stage2_width, kernel_size=3, stride=1, padding=1),
            nn.PReLU(stage2_width),
            nn.Conv2d(stage2_width, in_nc, kernel_size=1, stride=1, padding=0),
        )
        self.hr_refine = nn.Conv2d(in_nc, in_nc, kernel_size=1, stride=1, padding=0)
        arch_util.initialize_weights([self.stage1, self.stage2, self.hr_refine], 0.1)

    def forward(self, x, feat_lr):
        mr_base = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        feat_mr = F.interpolate(feat_lr, scale_factor=2, mode="bilinear", align_corners=False)

        delta1 = self.stage1(feat_mr)
        mr1 = mr_base + delta1

        delta2 = self.stage2(mr1)
        mr2 = mr1 + delta2

        hr0 = F.interpolate(mr2, scale_factor=2, mode="bilinear", align_corners=False)
        return hr0 + self.hr_refine(hr0)


class HardMRHeadV3MidA(nn.Module):
    """Two-stage MR correction with compact feature-guided stage2."""

    def __init__(self, in_nc=3, d=36, stage1_width=40, hint_width=8, stage2_width=24):
        super(HardMRHeadV3MidA, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(d, stage1_width, kernel_size=1, stride=1, padding=0),
            nn.PReLU(stage1_width),
            nn.Conv2d(stage1_width, stage1_width, kernel_size=3, stride=1, padding=1),
            nn.PReLU(stage1_width),
            nn.Conv2d(stage1_width, in_nc, kernel_size=1, stride=1, padding=0),
        )
        self.feat_hint = nn.Conv2d(d, hint_width, kernel_size=1, stride=1, padding=0)
        self.mr_encode = nn.Sequential(
            nn.Conv2d(in_nc, hint_width, kernel_size=3, stride=1, padding=1),
            nn.PReLU(hint_width),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(hint_width * 2, stage2_width, kernel_size=1, stride=1, padding=0),
            nn.PReLU(stage2_width),
            nn.Conv2d(stage2_width, stage2_width, kernel_size=3, stride=1, padding=1),
            nn.PReLU(stage2_width),
            nn.Conv2d(stage2_width, in_nc, kernel_size=1, stride=1, padding=0),
        )
        self.hr_refine = nn.Conv2d(in_nc, in_nc, kernel_size=1, stride=1, padding=0)
        arch_util.initialize_weights([self.stage1, self.feat_hint, self.mr_encode, self.stage2, self.hr_refine], 0.1)

    def forward(self, x, feat_lr):
        mr_base = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        feat_mr = F.interpolate(feat_lr, scale_factor=2, mode="bilinear", align_corners=False)

        delta1 = self.stage1(feat_mr)
        mr1 = mr_base + delta1

        feat_hint = self.feat_hint(feat_mr)
        mr1_feat = self.mr_encode(mr1)
        fused = torch.cat([feat_hint, mr1_feat], dim=1)
        delta2 = self.stage2(fused)
        mr2 = mr1 + delta2

        hr0 = F.interpolate(mr2, scale_factor=2, mode="bilinear", align_corners=False)
        return hr0 + self.hr_refine(hr0)


class HardMRHeadV4AddonA(nn.Module):
    """Medium-base MR path plus tiny feature-guided add-on for hard patches."""

    def __init__(self, in_nc=3, d=36, hint_width=4, stage2_width=12):
        super(HardMRHeadV4AddonA, self).__init__()
        self.mr_residual = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1),
            nn.PReLU(d),
            nn.Conv2d(d, in_nc, kernel_size=1, stride=1, padding=0),
        )
        self.feat_hint = nn.Conv2d(d, hint_width, kernel_size=1, stride=1, padding=0)
        self.mr_encode = nn.Sequential(
            nn.Conv2d(in_nc, hint_width, kernel_size=3, stride=1, padding=1),
            nn.PReLU(hint_width),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(hint_width * 2, stage2_width, kernel_size=1, stride=1, padding=0),
            nn.PReLU(stage2_width),
            nn.Conv2d(stage2_width, stage2_width, kernel_size=3, stride=1, padding=1),
            nn.PReLU(stage2_width),
            nn.Conv2d(stage2_width, in_nc, kernel_size=1, stride=1, padding=0),
        )
        self.hr_refine = nn.Conv2d(in_nc, in_nc, kernel_size=1, stride=1, padding=0)
        arch_util.initialize_weights([self.mr_residual, self.feat_hint, self.mr_encode, self.stage2, self.hr_refine], 0.1)

    def forward(self, x, feat_lr):
        mr_base = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        feat_mr = F.interpolate(feat_lr, scale_factor=2, mode="bilinear", align_corners=False)

        delta1 = self.mr_residual(feat_mr)
        mr1 = mr_base + delta1

        feat_hint = self.feat_hint(feat_mr)
        mr1_feat = self.mr_encode(mr1)
        fused = torch.cat([feat_hint, mr1_feat], dim=1)
        delta2 = self.stage2(fused)
        mr2 = mr1 + delta2

        hr0 = F.interpolate(mr2, scale_factor=2, mode="bilinear", align_corners=False)
        return hr0 + self.hr_refine(hr0)


class HardBranchAdapter(nn.Module):
    """Tiny hard-only feature adapter after the shared LR trunk."""

    def __init__(self, d=36):
        super(HardBranchAdapter, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=1, stride=1, padding=0),
            nn.PReLU(d),
        )
        # Start close to an identity passthrough so V4 weights remain usable.
        nn.init.dirac_(self.body[0].weight)
        if self.body[0].bias is not None:
            nn.init.constant_(self.body[0].bias, 0.0)
        nn.init.constant_(self.body[1].weight, 1.0)

    def forward(self, feat):
        return self.body(feat)


class fsrcnn_macinterp_branch1_net(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, hidden=8):
        super(fsrcnn_macinterp_branch1_net, self).__init__()
        self.head_easy = EasyInterpHead(in_nc=in_nc, out_nc=out_nc, hidden=hidden)

    def forward(self, x):
        return self.head_easy(x, None)


class fsrcnn_macinterp_branch1_lrhint_net(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, hidden=8, hint_hidden=8):
        super(fsrcnn_macinterp_branch1_lrhint_net, self).__init__()
        self.head_easy = EasyInterpHeadLRHint(
            in_nc=in_nc, out_nc=out_nc, hidden=hidden, hint_hidden=hint_hidden
        )

    def forward(self, x):
        return self.head_easy(x, None)


class fsrcnn_macinterp_branch2_net(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, d=36, s=12, m=4):
        super(fsrcnn_macinterp_branch2_net, self).__init__()
        self.shared_trunk = SharedLRTrunk(input_channels=in_nc, d=d, s=s, m=m)
        self.head_medium = MediumMRHead(in_nc=out_nc, d=d)

    def forward(self, x):
        feat = self.shared_trunk(x)
        return self.head_medium(x, feat)


class fsrcnn_medium_x4interp_3x3_branch2_net(FSRCNNInterp4xConv3xNet):
    def __init__(self, in_nc=3, out_nc=3, upscale=4, d=36, s=12, m=4):
        super(fsrcnn_medium_x4interp_3x3_branch2_net, self).__init__(
            input_channels=in_nc, upscale=upscale, d=d, s=s, m=m
        )


class fsrcnn_medium_2x3x3_2x3x3_branch2_net(FSRCNNInterp2x3x3Interp2x3x3Net):
    def __init__(self, in_nc=3, out_nc=3, upscale=4, d=36, s=12, m=4):
        super(fsrcnn_medium_2x3x3_2x3x3_branch2_net, self).__init__(
            input_channels=in_nc, upscale=upscale, d=d, s=s, m=m
        )


class fsrcnn_medium_2x3x3c24_2x3x3_branch2_net(FSRCNNInterp2x3x3C24Interp2x3x3Net):
    def __init__(self, in_nc=3, out_nc=3, upscale=4, d=36, s=12, m=4, mid_channels=24):
        super(fsrcnn_medium_2x3x3c24_2x3x3_branch2_net, self).__init__(
            input_channels=in_nc, upscale=upscale, d=d, s=s, m=m, mid_channels=mid_channels
        )


class fsrcnn_medium_2x3x3_2x1x1_branch2_net(FSRCNNInterp2x3x3Interp2x1x1Net):
    def __init__(self, in_nc=3, out_nc=3, upscale=4, d=36, s=12, m=4):
        super(fsrcnn_medium_2x3x3_2x1x1_branch2_net, self).__init__(
            input_channels=in_nc, upscale=upscale, d=d, s=s, m=m
        )


class fsrcnn_medium_bottleneck16_branch2_net(FSRCNNBottleneckMR2xNet):
    def __init__(self, in_nc=3, out_nc=3, upscale=4, d=36, s=12, m=4, bottleneck=16):
        super(fsrcnn_medium_bottleneck16_branch2_net, self).__init__(
            input_channels=in_nc, upscale=upscale, d=d, s=s, m=m, bottleneck=bottleneck
        )


class fsrcnn_macinterp_branch3_net(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, d=36, s=12, m=4):
        super(fsrcnn_macinterp_branch3_net, self).__init__()
        self.shared_trunk = SharedLRTrunk(input_channels=in_nc, d=d, s=s, m=m)
        self.head_hard = HardMRHead(in_nc=out_nc, d=d)

    def forward(self, x):
        feat = self.shared_trunk(x)
        return self.head_hard(x, feat)


class fsrcnn_macinterp_branch3_v2_net(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, d=36, s=12, m=4):
        super(fsrcnn_macinterp_branch3_v2_net, self).__init__()
        self.shared_trunk = SharedLRTrunk(input_channels=in_nc, d=d, s=s, m=m)
        self.head_hard = HardMRHeadV2(in_nc=out_nc, d=d)

    def forward(self, x):
        feat = self.shared_trunk(x)
        return self.head_hard(x, feat)


class fsrcnn_macinterp_branch3_v3_lite_net(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, d=36, s=12, m=4):
        super(fsrcnn_macinterp_branch3_v3_lite_net, self).__init__()
        self.shared_trunk = SharedLRTrunk(input_channels=in_nc, d=d, s=s, m=m)
        self.head_hard = HardMRHeadV3Lite(in_nc=out_nc, d=d)

    def forward(self, x):
        feat = self.shared_trunk(x)
        return self.head_hard(x, feat)


class fsrcnn_macinterp_branch3_v3_mid_a_net(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, d=36, s=12, m=4):
        super(fsrcnn_macinterp_branch3_v3_mid_a_net, self).__init__()
        self.shared_trunk = SharedLRTrunk(input_channels=in_nc, d=d, s=s, m=m)
        self.head_hard = HardMRHeadV3MidA(in_nc=out_nc, d=d)

    def forward(self, x):
        feat = self.shared_trunk(x)
        return self.head_hard(x, feat)


class fsrcnn_macinterp_branch3_v4_addon_a_net(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, d=36, s=12, m=4):
        super(fsrcnn_macinterp_branch3_v4_addon_a_net, self).__init__()
        self.shared_trunk = SharedLRTrunk(input_channels=in_nc, d=d, s=s, m=m)
        self.head_hard = HardMRHeadV4AddonA(in_nc=out_nc, d=d)

    def forward(self, x):
        feat = self.shared_trunk(x)
        return self.head_hard(x, feat)


class _ClassSR3Base(nn.Module):
    def __init__(self):
        super(_ClassSR3Base, self).__init__()
        self.classifier = Classifier()

    def forward(self, x, is_train):
        if is_train:
            for i in range(len(x)):
                type_out = self.classifier(x[i].unsqueeze(0))
                p = F.softmax(type_out, dim=1)
                p1 = p[0][0]
                p2 = p[0][1]
                p3 = p[0][2]

                out1 = self.net1(x[i].unsqueeze(0))
                out2 = self.net2(x[i].unsqueeze(0))
                out3 = self.net3(x[i].unsqueeze(0))
                out = out1 * p1 + out2 * p2 + out3 * p3

                if i == 0:
                    out_res = out
                    type_res = p
                else:
                    out_res = torch.cat((out_res, out), 0)
                    type_res = torch.cat((type_res, p), 0)
            return out_res, type_res

        for i in range(len(x)):
            type_out = self.classifier(x[i].unsqueeze(0))
            flag = torch.max(type_out, 1)[1].data.squeeze()
            p = F.softmax(type_out, dim=1)

            if flag == 0:
                out = self.net1(x[i].unsqueeze(0))
            elif flag == 1:
                out = self.net2(x[i].unsqueeze(0))
            else:
                out = self.net3(x[i].unsqueeze(0))

            if i == 0:
                out_res = out
                type_res = p
            else:
                out_res = torch.cat((out_res, out), 0)
                type_res = torch.cat((type_res, p), 0)

        return out_res, type_res


class classSR_3class_fsrcnn_net(_ClassSR3Base):
    """Baseline ClassSR-FSRCNN (all 3 branches use original FSRCNN tail)."""

    def __init__(self, in_nc=3, out_nc=3):
        super(classSR_3class_fsrcnn_net, self).__init__()
        self.upscale = 4
        self.net1 = FSRCNN_net(in_nc, self.upscale, 16, 12, 4)
        self.net2 = FSRCNN_net(in_nc, self.upscale, 36, 12, 4)
        self.net3 = FSRCNN_net(in_nc, self.upscale, 56, 12, 4)


class classSR_3class_fsrcnn_newtail_net(_ClassSR3Base):
    """
    Proposed mixed tails:
      - net1: x4 interp -> 1x1 -> 3x3
      - net2: FSRCNN body + x2 interp -> 3x3 -> x2 interp -> 1x1
      - net3: FSRCNN body + x2 interp -> 1x1 -> 3x3 -> x2 interp -> 1x1
    """

    def __init__(self, in_nc=3, out_nc=3):
        super(classSR_3class_fsrcnn_newtail_net, self).__init__()
        self.upscale = 4
        self.net1 = SimpleInterpCompNet(in_nc=in_nc, out_nc=out_nc, upscale=self.upscale, hidden=8)
        self.net2 = FSRCNNInterp2xComp2xNet(in_nc, self.upscale, 36, 12, 4)
        self.net3 = FSRCNNInterp2xPWComp2xNet(in_nc, self.upscale, 56, 12, 4)


class classSR_3class_fsrcnn_medium_x4interp_3x3_originhard_net(_ClassSR3Base):
    """
    SRAM-CIM ablation A:
      - easy: current interpolation residual head
      - medium: FSRCNN d=36 body + x4 feature interpolation + HR 3x3 36->3
      - hard: original ClassSR-FSRCNN hard branch (d=56, 9x9 stride-4 deconv)
    """

    def __init__(self, in_nc=3, out_nc=3):
        super(classSR_3class_fsrcnn_medium_x4interp_3x3_originhard_net, self).__init__()
        self.upscale = 4
        self.net1 = fsrcnn_macinterp_branch1_net(in_nc=in_nc, out_nc=out_nc, hidden=8)
        self.net2 = fsrcnn_medium_x4interp_3x3_branch2_net(
            in_nc=in_nc, out_nc=out_nc, upscale=self.upscale, d=36, s=12, m=4
        )
        self.net3 = FSRCNN_net(in_nc, self.upscale, 56, 12, 4)

    def _forward_all(self, x):
        prob = F.softmax(self.classifier(x), dim=1)
        out1 = self.net1(x)
        out2 = self.net2(x)
        out3 = self.net3(x)
        return prob, out1, out2, out3


class classSR_3class_fsrcnn_medium_2x3x3_2x3x3_originhard_net(_ClassSR3Base):
    """
    SRAM-CIM ablation B:
      - easy: current interpolation residual head
      - medium: FSRCNN d=36 body + x2 -> 3x3 36->36 -> x2 -> HR 3x3 36->3
      - hard: original ClassSR-FSRCNN hard branch (d=56, 9x9 stride-4 deconv)
    """

    def __init__(self, in_nc=3, out_nc=3):
        super(classSR_3class_fsrcnn_medium_2x3x3_2x3x3_originhard_net, self).__init__()
        self.upscale = 4
        self.net1 = fsrcnn_macinterp_branch1_net(in_nc=in_nc, out_nc=out_nc, hidden=8)
        self.net2 = fsrcnn_medium_2x3x3_2x3x3_branch2_net(
            in_nc=in_nc, out_nc=out_nc, upscale=self.upscale, d=36, s=12, m=4
        )
        self.net3 = FSRCNN_net(in_nc, self.upscale, 56, 12, 4)

    def _forward_all(self, x):
        prob = F.softmax(self.classifier(x), dim=1)
        out1 = self.net1(x)
        out2 = self.net2(x)
        out3 = self.net3(x)
        return prob, out1, out2, out3


class classSR_3class_fsrcnn_medium_bottleneck16_originhard_net(_ClassSR3Base):
    """
    SRAM-CIM ablation C:
      - easy: current interpolation residual head
      - medium: FSRCNN d=36 body + x2 -> 1x1 36->16 -> 3x3 16->16 -> 1x1 16->3 -> x2
      - hard: original ClassSR-FSRCNN hard branch (d=56, 9x9 stride-4 deconv)
    """

    def __init__(self, in_nc=3, out_nc=3):
        super(classSR_3class_fsrcnn_medium_bottleneck16_originhard_net, self).__init__()
        self.upscale = 4
        self.net1 = fsrcnn_macinterp_branch1_net(in_nc=in_nc, out_nc=out_nc, hidden=8)
        self.net2 = fsrcnn_medium_bottleneck16_branch2_net(
            in_nc=in_nc, out_nc=out_nc, upscale=self.upscale, d=36, s=12, m=4, bottleneck=16
        )
        self.net3 = FSRCNN_net(in_nc, self.upscale, 56, 12, 4)

    def _forward_all(self, x):
        prob = F.softmax(self.classifier(x), dim=1)
        out1 = self.net1(x)
        out2 = self.net2(x)
        out3 = self.net3(x)
        return prob, out1, out2, out3


class classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_net(_ClassSR3Base):
    """
    SRAM-CIM ablation D / version A:
      - easy: current interpolation residual head
      - medium: FSRCNN d=36 body + x2 -> 3x3 36->24 -> x2 -> HR 3x3 24->3
      - hard: original ClassSR-FSRCNN hard branch (d=56, 9x9 stride-4 deconv)
    """

    def __init__(self, in_nc=3, out_nc=3):
        super(classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_net, self).__init__()
        self.upscale = 4
        self.net1 = fsrcnn_macinterp_branch1_net(in_nc=in_nc, out_nc=out_nc, hidden=8)
        self.net2 = fsrcnn_medium_2x3x3c24_2x3x3_branch2_net(
            in_nc=in_nc, out_nc=out_nc, upscale=self.upscale, d=36, s=12, m=4, mid_channels=24
        )
        self.net3 = FSRCNN_net(in_nc, self.upscale, 56, 12, 4)

    def _forward_all(self, x):
        prob = F.softmax(self.classifier(x), dim=1)
        out1 = self.net1(x)
        out2 = self.net2(x)
        out3 = self.net3(x)
        return prob, out1, out2, out3


class classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_tinycnn_router_net(_ClassSR3Base):
    """
    Version A SR branches with a TinyCNN on-chip router.
    SR branches are identical to classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_net.
    """

    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        router_width=16,
        router_mid_ch=8,
        router_activation="relu",
        router_use_mid_projection=True,
    ):
        super(classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_tinycnn_router_net, self).__init__()
        self.classifier = TinyCNNRouter(
            in_ch=in_nc,
            width=router_width,
            mid_ch=router_mid_ch,
            num_classes=3,
            activation=router_activation,
            use_mid_projection=router_use_mid_projection,
        )
        self.upscale = 4
        self.net1 = fsrcnn_macinterp_branch1_net(in_nc=in_nc, out_nc=out_nc, hidden=8)
        self.net2 = fsrcnn_medium_2x3x3c24_2x3x3_branch2_net(
            in_nc=in_nc, out_nc=out_nc, upscale=self.upscale, d=36, s=12, m=4, mid_channels=24
        )
        self.net3 = FSRCNN_net(in_nc, self.upscale, 56, 12, 4)

    def _forward_all(self, x):
        prob = F.softmax(self.classifier(x), dim=1)
        out1 = self.net1(x)
        out2 = self.net2(x)
        out3 = self.net3(x)
        return prob, out1, out2, out3


class classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_tinycnn16_router_net(
    classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_tinycnn_router_net
):
    def __init__(self, in_nc=3, out_nc=3):
        super(classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_tinycnn16_router_net, self).__init__(
            in_nc=in_nc, out_nc=out_nc, router_width=16, router_mid_ch=8
        )


class classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_tinycnn8_router_net(
    classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_tinycnn_router_net
):
    def __init__(self, in_nc=3, out_nc=3):
        super(classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_tinycnn8_router_net, self).__init__(
            in_nc=in_nc, out_nc=out_nc, router_width=8, router_mid_ch=8, router_use_mid_projection=False
        )


class classSR_3class_fsrcnn_medium_2x3x3_2x1x1_originhard_net(_ClassSR3Base):
    """
    SRAM-CIM ablation E / version B:
      - easy: current interpolation residual head
      - medium: FSRCNN d=36 body + x2 -> 3x3 36->36 -> x2 -> HR 1x1 36->3
      - hard: original ClassSR-FSRCNN hard branch (d=56, 9x9 stride-4 deconv)
    """

    def __init__(self, in_nc=3, out_nc=3):
        super(classSR_3class_fsrcnn_medium_2x3x3_2x1x1_originhard_net, self).__init__()
        self.upscale = 4
        self.net1 = fsrcnn_macinterp_branch1_net(in_nc=in_nc, out_nc=out_nc, hidden=8)
        self.net2 = fsrcnn_medium_2x3x3_2x1x1_branch2_net(
            in_nc=in_nc, out_nc=out_nc, upscale=self.upscale, d=36, s=12, m=4
        )
        self.net3 = FSRCNN_net(in_nc, self.upscale, 56, 12, 4)

    def _forward_all(self, x):
        prob = F.softmax(self.classifier(x), dim=1)
        out1 = self.net1(x)
        out2 = self.net2(x)
        out3 = self.net3(x)
        return prob, out1, out2, out3


class classSR_3class_fsrcnn_macinterp_net(nn.Module):
    """
    Shared-trunk MAC+Interpolation co-designed ClassSR-FSRCNN.

    - Shared trunk is computed once in LR.
    - Easy head uses interpolation-only baseline with tiny HR correction.
    - Medium/Hard heads perform most learnable correction in MR.
    - Training uses soft routing; testing uses correctness-first hard routing
      while still materializing all three heads for stable validation.
    """

    def __init__(self, in_nc=3, out_nc=3, d=36, s=12, m=4, easy_hidden=8):
        super(classSR_3class_fsrcnn_macinterp_net, self).__init__()
        self.classifier = Classifier()
        self.shared_trunk = SharedLRTrunk(input_channels=in_nc, d=d, s=s, m=m)
        self.head_easy = EasyInterpHead(in_nc=in_nc, out_nc=out_nc, hidden=easy_hidden)
        self.head_medium = MediumMRHead(in_nc=out_nc, d=d)
        self.head_hard = HardMRHead(in_nc=out_nc, d=d)

    def _forward_all(self, x):
        prob = F.softmax(self.classifier(x), dim=1)
        feat = self.shared_trunk(x)
        out1 = self.head_easy(x, feat)
        out2 = self.head_medium(x, feat)
        out3 = self.head_hard(x, feat)
        return prob, out1, out2, out3

    def forward(self, x, is_train):
        prob, out1, out2, out3 = self._forward_all(x)
        self.last_branch_outputs = [out1, out2, out3]

        if is_train:
            fused = (
                out1 * prob[:, 0:1, None, None]
                + out2 * prob[:, 1:2, None, None]
                + out3 * prob[:, 2:3, None, None]
            )
            return fused, prob

        flags = torch.argmax(prob, dim=1)
        out = out1.clone()
        out[flags == 1] = out2[flags == 1]
        out[flags == 2] = out3[flags == 2]
        return out, prob


class classSR_3class_fsrcnn_macinterp_v2_net(nn.Module):
    """
    MACInterp v2: shared trunk / easy / medium / classifier unchanged;
    only the hard head is upgraded to two-stage MR residual correction.
    """

    def __init__(self, in_nc=3, out_nc=3, d=36, s=12, m=4, easy_hidden=8):
        super(classSR_3class_fsrcnn_macinterp_v2_net, self).__init__()
        self.classifier = Classifier()
        self.shared_trunk = SharedLRTrunk(input_channels=in_nc, d=d, s=s, m=m)
        self.head_easy = EasyInterpHead(in_nc=in_nc, out_nc=out_nc, hidden=easy_hidden)
        self.head_medium = MediumMRHead(in_nc=out_nc, d=d)
        self.head_hard = HardMRHeadV2(in_nc=out_nc, d=d)

    def _forward_all(self, x):
        prob = F.softmax(self.classifier(x), dim=1)
        feat = self.shared_trunk(x)
        out1 = self.head_easy(x, feat)
        out2 = self.head_medium(x, feat)
        out3 = self.head_hard(x, feat)
        return prob, out1, out2, out3

    def forward(self, x, is_train):
        prob, out1, out2, out3 = self._forward_all(x)
        self.last_branch_outputs = [out1, out2, out3]

        if is_train:
            fused = (
                out1 * prob[:, 0:1, None, None]
                + out2 * prob[:, 1:2, None, None]
                + out3 * prob[:, 2:3, None, None]
            )
            return fused, prob

        flags = torch.argmax(prob, dim=1)
        out = out1.clone()
        out[flags == 1] = out2[flags == 1]
        out[flags == 2] = out3[flags == 2]
        return out, prob


class classSR_3class_fsrcnn_macinterp_v3_lite_net(nn.Module):
    """
    MACInterp v3-lite: shared trunk / easy / medium / classifier unchanged;
    hard head keeps two-stage MR correction but uses image-driven lightweight stage2.
    """

    def __init__(self, in_nc=3, out_nc=3, d=36, s=12, m=4, easy_hidden=8):
        super(classSR_3class_fsrcnn_macinterp_v3_lite_net, self).__init__()
        self.classifier = Classifier()
        self.shared_trunk = SharedLRTrunk(input_channels=in_nc, d=d, s=s, m=m)
        self.head_easy = EasyInterpHead(in_nc=in_nc, out_nc=out_nc, hidden=easy_hidden)
        self.head_medium = MediumMRHead(in_nc=out_nc, d=d)
        self.head_hard = HardMRHeadV3Lite(in_nc=out_nc, d=d)

    def _forward_all(self, x):
        prob = F.softmax(self.classifier(x), dim=1)
        feat = self.shared_trunk(x)
        out1 = self.head_easy(x, feat)
        out2 = self.head_medium(x, feat)
        out3 = self.head_hard(x, feat)
        return prob, out1, out2, out3

    def forward(self, x, is_train):
        prob, out1, out2, out3 = self._forward_all(x)
        self.last_branch_outputs = [out1, out2, out3]

        if is_train:
            fused = (
                out1 * prob[:, 0:1, None, None]
                + out2 * prob[:, 1:2, None, None]
                + out3 * prob[:, 2:3, None, None]
            )
            return fused, prob

        flags = torch.argmax(prob, dim=1)
        out = out1.clone()
        out[flags == 1] = out2[flags == 1]
        out[flags == 2] = out3[flags == 2]
        return out, prob


class classSR_3class_fsrcnn_macinterp_v3_mid_a_net(nn.Module):
    """
    MACInterp v3-mid-a: shared trunk / easy / medium / classifier unchanged;
    hard head uses compact feature-guided stage2 at MR scale.
    """

    def __init__(self, in_nc=3, out_nc=3, d=36, s=12, m=4, easy_hidden=8):
        super(classSR_3class_fsrcnn_macinterp_v3_mid_a_net, self).__init__()
        self.classifier = Classifier()
        self.shared_trunk = SharedLRTrunk(input_channels=in_nc, d=d, s=s, m=m)
        self.head_easy = EasyInterpHead(in_nc=in_nc, out_nc=out_nc, hidden=easy_hidden)
        self.head_medium = MediumMRHead(in_nc=out_nc, d=d)
        self.head_hard = HardMRHeadV3MidA(in_nc=out_nc, d=d)

    def _forward_all(self, x):
        prob = F.softmax(self.classifier(x), dim=1)
        feat = self.shared_trunk(x)
        out1 = self.head_easy(x, feat)
        out2 = self.head_medium(x, feat)
        out3 = self.head_hard(x, feat)
        return prob, out1, out2, out3

    def forward(self, x, is_train):
        prob, out1, out2, out3 = self._forward_all(x)
        self.last_branch_outputs = [out1, out2, out3]

        if is_train:
            fused = (
                out1 * prob[:, 0:1, None, None]
                + out2 * prob[:, 1:2, None, None]
                + out3 * prob[:, 2:3, None, None]
            )
            return fused, prob

        flags = torch.argmax(prob, dim=1)
        out = out1.clone()
        out[flags == 1] = out2[flags == 1]
        out[flags == 2] = out3[flags == 2]
        return out, prob


class classSR_3class_fsrcnn_macinterp_v4_addon_a_net(nn.Module):
    """
    MACInterp v4-addon-a: medium-base hard stage1 plus tiny feature-guided add-on.
    """

    def __init__(self, in_nc=3, out_nc=3, d=36, s=12, m=4, easy_hidden=8):
        super(classSR_3class_fsrcnn_macinterp_v4_addon_a_net, self).__init__()
        self.classifier = Classifier()
        self.shared_trunk = SharedLRTrunk(input_channels=in_nc, d=d, s=s, m=m)
        self.head_easy = EasyInterpHead(in_nc=in_nc, out_nc=out_nc, hidden=easy_hidden)
        self.head_medium = MediumMRHead(in_nc=out_nc, d=d)
        self.head_hard = HardMRHeadV4AddonA(in_nc=out_nc, d=d)

    def _forward_all(self, x):
        prob = F.softmax(self.classifier(x), dim=1)
        feat = self.shared_trunk(x)
        out1 = self.head_easy(x, feat)
        out2 = self.head_medium(x, feat)
        out3 = self.head_hard(x, feat)
        return prob, out1, out2, out3

    def forward(self, x, is_train):
        prob, out1, out2, out3 = self._forward_all(x)
        self.last_branch_outputs = [out1, out2, out3]

        if is_train:
            fused = (
                out1 * prob[:, 0:1, None, None]
                + out2 * prob[:, 1:2, None, None]
                + out3 * prob[:, 2:3, None, None]
            )
            return fused, prob

        flags = torch.argmax(prob, dim=1)
        out = out1.clone()
        out[flags == 1] = out2[flags == 1]
        out[flags == 2] = out3[flags == 2]
        return out, prob


class classSR_3class_fsrcnn_macinterp_v4_addon_a_adapter_net(nn.Module):
    """
    MACInterp v4-addon-a + tiny hard-only adapter after the shared trunk.
    """

    def __init__(self, in_nc=3, out_nc=3, d=36, s=12, m=4, easy_hidden=8):
        super(classSR_3class_fsrcnn_macinterp_v4_addon_a_adapter_net, self).__init__()
        self.classifier = Classifier()
        self.shared_trunk = SharedLRTrunk(input_channels=in_nc, d=d, s=s, m=m)
        self.head_easy = EasyInterpHead(in_nc=in_nc, out_nc=out_nc, hidden=easy_hidden)
        self.head_medium = MediumMRHead(in_nc=out_nc, d=d)
        self.hard_adapter = HardBranchAdapter(d=d)
        self.head_hard = HardMRHeadV4AddonA(in_nc=out_nc, d=d)

    def _forward_all(self, x):
        prob = F.softmax(self.classifier(x), dim=1)
        feat = self.shared_trunk(x)
        out1 = self.head_easy(x, feat)
        out2 = self.head_medium(x, feat)
        out3 = self.head_hard(x, self.hard_adapter(feat))
        return prob, out1, out2, out3

    def forward(self, x, is_train):
        prob, out1, out2, out3 = self._forward_all(x)
        self.last_branch_outputs = [out1, out2, out3]

        if is_train:
            fused = (
                out1 * prob[:, 0:1, None, None]
                + out2 * prob[:, 1:2, None, None]
                + out3 * prob[:, 2:3, None, None]
            )
            return fused, prob

        flags = torch.argmax(prob, dim=1)
        out = out1.clone()
        out[flags == 1] = out2[flags == 1]
        out[flags == 2] = out3[flags == 2]
        return out, prob


class classSR_3class_fsrcnn_macinterp_v4_addon_a_adapter_easylrhint_net(nn.Module):
    """
    MACInterp v4-addon-a + tiny hard-only adapter, with LR-hint easy branch.
    """

    def __init__(self, in_nc=3, out_nc=3, d=36, s=12, m=4, easy_hidden=8, hint_hidden=8):
        super(classSR_3class_fsrcnn_macinterp_v4_addon_a_adapter_easylrhint_net, self).__init__()
        self.classifier = Classifier()
        self.shared_trunk = SharedLRTrunk(input_channels=in_nc, d=d, s=s, m=m)
        self.head_easy = EasyInterpHeadLRHint(
            in_nc=in_nc, out_nc=out_nc, hidden=easy_hidden, hint_hidden=hint_hidden
        )
        self.head_medium = MediumMRHead(in_nc=out_nc, d=d)
        self.hard_adapter = HardBranchAdapter(d=d)
        self.head_hard = HardMRHeadV4AddonA(in_nc=out_nc, d=d)

    def _forward_all(self, x):
        prob = F.softmax(self.classifier(x), dim=1)
        feat = self.shared_trunk(x)
        out1 = self.head_easy(x, feat)
        out2 = self.head_medium(x, feat)
        out3 = self.head_hard(x, self.hard_adapter(feat))
        return prob, out1, out2, out3

    def forward(self, x, is_train):
        prob, out1, out2, out3 = self._forward_all(x)
        self.last_branch_outputs = [out1, out2, out3]

        if is_train:
            fused = (
                out1 * prob[:, 0:1, None, None]
                + out2 * prob[:, 1:2, None, None]
                + out3 * prob[:, 2:3, None, None]
            )
            return fused, prob

        flags = torch.argmax(prob, dim=1)
        out = out1.clone()
        out[flags == 1] = out2[flags == 1]
        out[flags == 2] = out3[flags == 2]
        return out, prob
