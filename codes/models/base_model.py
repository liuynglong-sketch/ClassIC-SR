import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup
        lr_groups_l: list for lr_groups. each for a optimizer"""
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler"""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        # set up warm-up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        return str(network), sum(map(lambda x: x.numel(), network.parameters()))

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def _filter_state_dict(self, load_net_clean, network):
        net_state = network.state_dict()
        filtered = OrderedDict()
        for k, v in load_net_clean.items():
            if k in net_state and net_state[k].shape == v.shape:
                filtered[k] = v
        return filtered

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        if not strict:
            load_net_clean = self._filter_state_dict(load_net_clean, network)
        network.load_state_dict(load_net_clean, strict=strict)

    def load_network_classifier(self,load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module.classifier
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def load_network_classifier_rcan(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network1 = network.module.classifier
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('classifier'):

                load_net_clean[k[11:]] = v
            else:
                pass
        network1.load_state_dict(load_net_clean, strict=strict)

    def load_network_classifier_(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network1 = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('classifier'):

                load_net_clean[k[11:]] = v
            else:
                pass
        network1.load_state_dict(load_net_clean, strict=strict)

    def load_network_classSR_2class(self,load_path, network, strict=True):

        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network1 = network.module.net1
            network2 = network.module.net2
        load_net = torch.load(load_path[0])
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network1.load_state_dict(load_net_clean, strict=strict)

        load_net = torch.load(load_path[1])
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network2.load_state_dict(load_net_clean, strict=strict)

    def load_network_classSR_3class(self,load_path, network, strict=True):

        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network1 = network.module.net1
            network2 = network.module.net2
            network3 = network.module.net3
        load_net = torch.load(load_path[0])
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        if not strict:
            load_net_clean = self._filter_state_dict(load_net_clean, network1)
        network1.load_state_dict(load_net_clean, strict=strict)

        load_net = torch.load(load_path[1])
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        if not strict:
            load_net_clean = self._filter_state_dict(load_net_clean, network2)
        network2.load_state_dict(load_net_clean, strict=strict)

        load_net = torch.load(load_path[2])
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        if not strict:
            load_net_clean = self._filter_state_dict(load_net_clean, network3)
        network3.load_state_dict(load_net_clean, strict=strict)

    def _load_prefixed_submodule(self, load_path, submodule, prefix, strict=True):
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()
        for k, v in load_net.items():
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith(prefix):
                load_net_clean[k[len(prefix):]] = v
        if not strict:
            load_net_clean = self._filter_state_dict(load_net_clean, submodule)
        submodule.load_state_dict(load_net_clean, strict=strict)

    def load_network_classSR_macinterp_3class(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module

        # Initialize the shared trunk from branch2 by default, since branch2
        # sits closest to the intended shared-capacity operating point.
        self._load_prefixed_submodule(load_path[1], network.shared_trunk, 'shared_trunk.', strict=strict)
        self._load_prefixed_submodule(load_path[0], network.head_easy, 'head_easy.', strict=strict)
        self._load_prefixed_submodule(load_path[1], network.head_medium, 'head_medium.', strict=strict)
        self._load_prefixed_submodule(load_path[2], network.head_hard, 'head_hard.', strict=strict)

    def load_network_fsrcnn_trunk_into_macinterp(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module

        target = network.shared_trunk
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()
        for k, v in load_net.items():
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('head_conv.'):
                load_net_clean['head_conv.' + k[len('head_conv.'):]] = v
            elif k.startswith('body_conv.'):
                load_net_clean['body_conv.' + k[len('body_conv.'):]] = v

        if not strict:
            load_net_clean = self._filter_state_dict(load_net_clean, target)
        target.load_state_dict(load_net_clean, strict=strict)

    def load_network_branch2_into_macinterp_hard_v4(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module

        load_net = torch.load(load_path)
        for k in load_net.keys():
            key = k[7:] if k.startswith('module.') else k
            if key.startswith('head_hard.') or key.startswith('shared_trunk.'):
                if key.startswith('head_hard.'):
                    network.load_state_dict(self._filter_state_dict(OrderedDict(
                        (kk[7:] if kk.startswith('module.') else kk, vv) for kk, vv in load_net.items()
                    ), network), strict=False if not strict else True)
                    return

        trunk_state = OrderedDict()
        stage1_state = OrderedDict()
        hr_state = OrderedDict()
        for k, v in load_net.items():
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('shared_trunk.'):
                trunk_state[k[len('shared_trunk.'):]] = v
            elif k.startswith('head_medium.mr_residual.'):
                stage1_state['mr_residual.' + k[len('head_medium.mr_residual.'):]] = v
            elif k.startswith('head_medium.hr_refine.'):
                hr_state['hr_refine.' + k[len('head_medium.hr_refine.'):]] = v

        if not strict:
            trunk_state = self._filter_state_dict(trunk_state, network.shared_trunk)
            head_state = OrderedDict()
            head_state.update(stage1_state)
            head_state.update(hr_state)
            head_state = self._filter_state_dict(head_state, network.head_hard)
            network.shared_trunk.load_state_dict(trunk_state, strict=False)
            network.head_hard.load_state_dict(head_state, strict=False)
            return

        network.shared_trunk.load_state_dict(trunk_state, strict=True)
        head_state = OrderedDict()
        head_state.update(stage1_state)
        head_state.update(hr_state)
        network.head_hard.load_state_dict(head_state, strict=False)

    def load_network_classSR_4class(self,load_path, network, strict=True):

        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network1 = network.module.net1
            network2 = network.module.net2
            network3 = network.module.net3
            network4 = network.module.net4

        load_net = torch.load(load_path[0])
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network1.load_state_dict(load_net_clean, strict=strict)

        load_net = torch.load(load_path[1])
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network2.load_state_dict(load_net_clean, strict=strict)

        load_net = torch.load(load_path[2])
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network3.load_state_dict(load_net_clean, strict=strict)

        load_net = torch.load(load_path[3])
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network4.load_state_dict(load_net_clean, strict=strict)

    def load_network_classSR_5class(self,load_path, network, strict=True):

        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network1 = network.module.net1
            network2 = network.module.net2
            network3 = network.module.net3
            network4 = network.module.net4
            network5 = network.module.net5
        load_net = torch.load(load_path[0])
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network1.load_state_dict(load_net_clean, strict=strict)

        load_net = torch.load(load_path[1])
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network2.load_state_dict(load_net_clean, strict=strict)

        load_net = torch.load(load_path[2])
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network3.load_state_dict(load_net_clean, strict=strict)

        load_net = torch.load(load_path[3])
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network4.load_state_dict(load_net_clean, strict=strict)

        load_net = torch.load(load_path[4])
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network5.load_state_dict(load_net_clean, strict=strict)

    

    def save_training_state(self, epoch, iter_step):
        """Save training state during training, which will be used for resuming"""
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        """Resume the optimizers and schedulers for training"""
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
