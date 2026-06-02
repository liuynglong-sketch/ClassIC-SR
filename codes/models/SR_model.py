import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss
from torchsummary import summary
import utils.util as util
import time


logger = logging.getLogger('base')


class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)



        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.netTeacher = None


        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()
            freeze_modules = train_opt.get('freeze_modules')
            if freeze_modules:
                self._freeze_named_modules(freeze_modules)

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']
            self.l_distill_w = train_opt.get('distill_weight', 0)

            teacher_path = self.opt['path'].get('pretrain_model_teacher_G')
            teacher_net_opt = self.opt.get('network_teacher_G')
            if self.l_distill_w > 0 and teacher_path is not None and teacher_net_opt is not None:
                teacher_opt = {'network_G': teacher_net_opt}
                self.netTeacher = networks.define_G(teacher_opt).to(self.device)
                self.load_network(teacher_path, self.netTeacher, True)
                self.netTeacher.eval()
                for p in self.netTeacher.parameters():
                    p.requires_grad = False
                logger.info('Teacher model loaded for distillation [{:s}] ...'.format(teacher_path))

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['T_period'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def _freeze_named_modules(self, module_prefixes):
        prefixes = tuple(module_prefixes)
        for name, param in self.netG.named_parameters():
            clean_name = name[7:] if name.startswith('module.') else name
            if clean_name.startswith(prefixes):
                param.requires_grad = False
                if self.rank <= 0:
                    logger.info('Freeze parameter [{:s}]'.format(clean_name))

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device, non_blocking=True)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device, non_blocking=True)  # GT

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_total = l_pix
        if self.netTeacher is not None:
            with torch.no_grad():
                teacher_H = self.netTeacher(self.var_L)
            l_distill = self.l_distill_w * self.cri_pix(self.fake_H, teacher_H.detach())
            l_total = l_total + l_distill
            self.log_dict['l_distill'] = l_distill.item()
        l_total.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            which_model = self.opt['network_G']['which_model_G']
            if which_model in [
                'fsrcnn_macinterp_branch2_net',
                'fsrcnn_macinterp_branch3_net',
                'fsrcnn_macinterp_branch3_v2_net',
                'fsrcnn_macinterp_branch3_v3_lite_net',
                'fsrcnn_macinterp_branch3_v3_mid_a_net',
                'fsrcnn_macinterp_branch3_v4_addon_a_net',
            ]:
                if which_model == 'fsrcnn_macinterp_branch3_v4_addon_a_net':
                    self.load_network_branch2_into_macinterp_hard_v4(load_path_G, self.netG, self.opt['path']['strict_load'])
                else:
                    self.load_network_fsrcnn_trunk_into_macinterp(load_path_G, self.netG, self.opt['path']['strict_load'])
            else:
                self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
