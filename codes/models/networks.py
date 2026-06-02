import torch
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.classSR_fsrcnn_arch as classSR_3class_arch
import models.archs.classSR_rcan_arch as classSR_rcan_arch
import models.archs.classSR_carn_arch as classSR_carn_arch
import models.archs.classSR_srresnet_arch as classSR_srresnet_arch
import models.archs.RCAN_arch as RCAN_arch
import models.archs.FSRCNN_arch as FSRCNN_arch
import models.archs.CARN_arch as CARN_arch
from models.archs import BilinearUpsample_arch, Bilinear_FSRCNN_arch


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'MSRResNetInterpTail':
        netG = SRResNet_arch.MSRResNetInterpTail(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                                 nf=opt_net['nf'], nb=opt_net['nb'],
                                                 upscale=opt_net['scale'],
                                                 comp_channels=opt_net.get('comp_channels', 32))

    elif which_model == 'RCAN':
        netG = RCAN_arch.RCAN(n_resblocks=opt_net['n_resblocks'], n_feats=opt_net['n_feats'],
                              res_scale=opt_net['res_scale'], n_colors=opt_net['n_colors'],rgb_range=opt_net['rgb_range'],
                              scale=opt_net['scale'],reduction=opt_net['reduction'],n_resgroups=opt_net['n_resgroups'])
    elif which_model == 'CARN_M':
        netG = CARN_arch.CARN_M(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], scale=opt_net['scale'], group=opt_net['group'])

    elif which_model == 'fsrcnn':
        netG = FSRCNN_arch.FSRCNN_net(input_channels=opt_net['in_nc'],upscale=opt_net['scale'],d=opt_net['d'],
                                      s=opt_net['s'],m=opt_net['m'])
### ----------------------------------------------------------------------------------------------------
    elif which_model == 'BilinearUpsample':
        netG = BilinearUpsample_arch.BilinearUpsample_net( upscale=opt_net['scale'])

    elif which_model == 'Bilinear_FSRCNN':
        netG = Bilinear_FSRCNN_arch.Bilinear_FSRCNN_net(input_channels=opt_net['in_nc'],upscale=opt_net['scale'],d=opt_net['d'],
                                      s=opt_net['s'],m=opt_net['m'])
    elif which_model == 'simple_interp_comp':
        netG = classSR_3class_arch.SimpleInterpCompNet(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], upscale=opt_net['scale'], hidden=opt_net.get('hidden', 8)
        )
    elif which_model == 'fsrcnn_interp2x_comp2x':
        netG = classSR_3class_arch.FSRCNNInterp2xComp2xNet(
            input_channels=opt_net['in_nc'], upscale=opt_net['scale'], d=opt_net['d'], s=opt_net['s'], m=opt_net['m']
        )
    elif which_model == 'fsrcnn_interp2x_pwcomp2x':
        netG = classSR_3class_arch.FSRCNNInterp2xPWComp2xNet(
            input_channels=opt_net['in_nc'], upscale=opt_net['scale'], d=opt_net['d'], s=opt_net['s'], m=opt_net['m']
        )
    elif which_model == 'fsrcnn_macinterp_branch1_net':
        netG = classSR_3class_arch.fsrcnn_macinterp_branch1_net(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], hidden=opt_net.get('hidden', 8)
        )
    elif which_model == 'fsrcnn_macinterp_branch1_lrhint_net':
        netG = classSR_3class_arch.fsrcnn_macinterp_branch1_lrhint_net(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            hidden=opt_net.get('hidden', 8),
            hint_hidden=opt_net.get('hint_hidden', 8),
        )
    elif which_model == 'fsrcnn_macinterp_branch2_net':
        netG = classSR_3class_arch.fsrcnn_macinterp_branch2_net(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], d=opt_net['d'], s=opt_net['s'], m=opt_net['m']
        )
    elif which_model == 'fsrcnn_medium_x4interp_3x3_branch2_net':
        netG = classSR_3class_arch.fsrcnn_medium_x4interp_3x3_branch2_net(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            upscale=opt_net.get('scale', 4),
            d=opt_net['d'],
            s=opt_net['s'],
            m=opt_net['m'],
        )
    elif which_model == 'fsrcnn_medium_2x3x3_2x3x3_branch2_net':
        netG = classSR_3class_arch.fsrcnn_medium_2x3x3_2x3x3_branch2_net(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            upscale=opt_net.get('scale', 4),
            d=opt_net['d'],
            s=opt_net['s'],
            m=opt_net['m'],
        )
    elif which_model == 'fsrcnn_medium_2x3x3c24_2x3x3_branch2_net':
        netG = classSR_3class_arch.fsrcnn_medium_2x3x3c24_2x3x3_branch2_net(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            upscale=opt_net.get('scale', 4),
            d=opt_net['d'],
            s=opt_net['s'],
            m=opt_net['m'],
            mid_channels=opt_net.get('mid_channels', 24),
        )
    elif which_model == 'fsrcnn_medium_2x3x3_2x1x1_branch2_net':
        netG = classSR_3class_arch.fsrcnn_medium_2x3x3_2x1x1_branch2_net(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            upscale=opt_net.get('scale', 4),
            d=opt_net['d'],
            s=opt_net['s'],
            m=opt_net['m'],
        )
    elif which_model == 'fsrcnn_medium_bottleneck16_branch2_net':
        netG = classSR_3class_arch.fsrcnn_medium_bottleneck16_branch2_net(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            upscale=opt_net.get('scale', 4),
            d=opt_net['d'],
            s=opt_net['s'],
            m=opt_net['m'],
            bottleneck=opt_net.get('bottleneck', 16),
        )
    elif which_model == 'fsrcnn_macinterp_branch3_net':
        netG = classSR_3class_arch.fsrcnn_macinterp_branch3_net(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], d=opt_net['d'], s=opt_net['s'], m=opt_net['m']
        )
    elif which_model == 'fsrcnn_macinterp_branch3_v2_net':
        netG = classSR_3class_arch.fsrcnn_macinterp_branch3_v2_net(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], d=opt_net['d'], s=opt_net['s'], m=opt_net['m']
        )
    elif which_model == 'fsrcnn_macinterp_branch3_v3_lite_net':
        netG = classSR_3class_arch.fsrcnn_macinterp_branch3_v3_lite_net(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], d=opt_net['d'], s=opt_net['s'], m=opt_net['m']
        )
    elif which_model == 'fsrcnn_macinterp_branch3_v3_mid_a_net':
        netG = classSR_3class_arch.fsrcnn_macinterp_branch3_v3_mid_a_net(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], d=opt_net['d'], s=opt_net['s'], m=opt_net['m']
        )
    elif which_model == 'fsrcnn_macinterp_branch3_v4_addon_a_net':
        netG = classSR_3class_arch.fsrcnn_macinterp_branch3_v4_addon_a_net(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], d=opt_net['d'], s=opt_net['s'], m=opt_net['m']
        )

 ### ----------------------------------------------------------------------------------------------------


    elif which_model == 'classSR_3class_fsrcnn_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_net(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'classSR_3class_fsrcnn_newtail_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_newtail_net(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc']
        )
    elif which_model == 'classSR_3class_fsrcnn_medium_x4interp_3x3_originhard_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_medium_x4interp_3x3_originhard_net(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc']
        )
    elif which_model == 'classSR_3class_fsrcnn_medium_2x3x3_2x3x3_originhard_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_medium_2x3x3_2x3x3_originhard_net(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc']
        )
    elif which_model == 'classSR_3class_fsrcnn_medium_bottleneck16_originhard_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_medium_bottleneck16_originhard_net(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc']
        )
    elif which_model == 'classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_net(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc']
        )
    elif which_model == 'classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_tinycnn_router_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_tinycnn_router_net(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            router_width=opt_net.get('router_width', 16),
            router_mid_ch=opt_net.get('router_mid_ch', 8),
            router_activation=opt_net.get('router_activation', 'relu'),
            router_use_mid_projection=opt_net.get('router_use_mid_projection', True),
        )
    elif which_model == 'classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_tinycnn16_router_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_tinycnn16_router_net(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc']
        )
    elif which_model == 'classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_tinycnn8_router_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_medium_2x3x3c24_2x3x3_originhard_tinycnn8_router_net(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc']
        )
    elif which_model == 'classSR_3class_fsrcnn_medium_2x3x3_2x1x1_originhard_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_medium_2x3x3_2x1x1_originhard_net(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc']
        )
    elif which_model == 'classSR_3class_fsrcnn_macinterp_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_macinterp_net(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            d=opt_net.get('d', 36),
            s=opt_net.get('s', 12),
            m=opt_net.get('m', 4),
            easy_hidden=opt_net.get('hidden', 8)
        )
    elif which_model == 'classSR_3class_fsrcnn_macinterp_v2_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_macinterp_v2_net(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            d=opt_net.get('d', 36),
            s=opt_net.get('s', 12),
            m=opt_net.get('m', 4),
            easy_hidden=opt_net.get('hidden', 8)
        )
    elif which_model == 'classSR_3class_fsrcnn_macinterp_v3_lite_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_macinterp_v3_lite_net(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            d=opt_net.get('d', 36),
            s=opt_net.get('s', 12),
            m=opt_net.get('m', 4),
            easy_hidden=opt_net.get('hidden', 8)
        )
    elif which_model == 'classSR_3class_fsrcnn_macinterp_v3_mid_a_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_macinterp_v3_mid_a_net(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            d=opt_net.get('d', 36),
            s=opt_net.get('s', 12),
            m=opt_net.get('m', 4),
            easy_hidden=opt_net.get('hidden', 8)
        )
    elif which_model == 'classSR_3class_fsrcnn_macinterp_v4_addon_a_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_macinterp_v4_addon_a_net(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            d=opt_net.get('d', 36),
            s=opt_net.get('s', 12),
            m=opt_net.get('m', 4),
            easy_hidden=opt_net.get('hidden', 8)
        )
    elif which_model == 'classSR_3class_fsrcnn_macinterp_v4_addon_a_adapter_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_macinterp_v4_addon_a_adapter_net(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            d=opt_net.get('d', 36),
            s=opt_net.get('s', 12),
            m=opt_net.get('m', 4),
            easy_hidden=opt_net.get('hidden', 8)
        )
    elif which_model == 'classSR_3class_fsrcnn_macinterp_v4_addon_a_adapter_easylrhint_net':
        netG = classSR_3class_arch.classSR_3class_fsrcnn_macinterp_v4_addon_a_adapter_easylrhint_net(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            d=opt_net.get('d', 36),
            s=opt_net.get('s', 12),
            m=opt_net.get('m', 4),
            easy_hidden=opt_net.get('hidden', 8),
            hint_hidden=opt_net.get('hint_hidden', 8)
        )
    elif which_model == 'classSR_3class_rcan':
        netG = classSR_rcan_arch.classSR_3class_rcan(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'classSR_3class_srresnet':
        netG = classSR_srresnet_arch.ClassSR(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'classSR_3class_srresnet_interp_tail_net':
        netG = classSR_srresnet_arch.ClassSRInterpTail(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'classSR_3class_carn':
        netG = classSR_carn_arch.ClassSR(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
