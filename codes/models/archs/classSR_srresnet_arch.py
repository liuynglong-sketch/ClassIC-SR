import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch
from models.archs.SRResNet_arch import MSRResNet, MSRResNetInterpTail
import numpy as np
import time

class ClassSR(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(ClassSR, self).__init__()
        self.upscale=4
        self.classifier=Classifier()
        self.net1 = MSRResNet(in_nc, out_nc, 36, 16, 4)
        self.net2 = MSRResNet(in_nc, out_nc, 52, 16, 4)
        self.net3 = MSRResNet(in_nc, out_nc, 64, 16, 4)

    def forward(self, x,is_train):
        if is_train:
            for i in range(len(x)):
                # print(x[i].unsqueeze(0).shape)
                type = self.classifier(x[i].unsqueeze(0))
                p = F.softmax(type, dim=1)

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
        else:

            for i in range(len(x)):
                type = self.classifier(x[i].unsqueeze(0))

                flag = torch.max(type, 1)[1].data.squeeze()
                p = F.softmax(type, dim=1)
                #flag=np.random.randint(0,2)
                #flag=2
                if flag == 0:
                    out = self.net1(x[i].unsqueeze(0))
                elif flag==1:
                    out = self.net2(x[i].unsqueeze(0))
                elif flag==2:
                    out = self.net3(x[i].unsqueeze(0))
                if i == 0:
                    out_res = out
                    type_res = p
                else:
                    out_res = torch.cat((out_res, out), 0)
                    type_res = torch.cat((type_res, p), 0)

            return out_res, type_res

        return out_res,type_res

class ClassSRInterpTail(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(ClassSRInterpTail, self).__init__()
        self.upscale=4
        self.classifier=Classifier()
        self.net1 = MSRResNetInterpTail(in_nc, out_nc, 36, 16, 4, comp_channels=16)
        self.net2 = MSRResNetInterpTail(in_nc, out_nc, 52, 16, 4, comp_channels=24)
        self.net3 = MSRResNetInterpTail(in_nc, out_nc, 64, 16, 4, comp_channels=32)

    def forward(self, x,is_train):
        if is_train:
            type_logits = self.classifier(x)
            type_res = F.softmax(type_logits, dim=1)

            out1 = self.net1(x)
            out2 = self.net2(x)
            out3 = self.net3(x)
            p = type_res.view(type_res.size(0), type_res.size(1), 1, 1, 1)
            out_res = out1 * p[:, 0] + out2 * p[:, 1] + out3 * p[:, 2]
        else:
            type_logits = self.classifier(x)
            type_res = F.softmax(type_logits, dim=1)
            flags = torch.max(type_logits, 1)[1]
            out_res = x.new_empty(x.size(0), 3, x.size(2) * self.upscale, x.size(3) * self.upscale)
            for flag, branch in ((0, self.net1), (1, self.net2), (2, self.net3)):
                idx = torch.nonzero(flags == flag, as_tuple=False).view(-1)
                if idx.numel() > 0:
                    out_res[idx] = branch(x.index_select(0, idx))

            return out_res, type_res

        return out_res,type_res

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lastOut = nn.Linear(32, 3)

        # Condtion network
        self.CondNet = nn.Sequential(nn.Conv2d(3, 128, 4, 4), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 32, 1))
        arch_util.initialize_weights([self.CondNet], 0.1)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AvgPool2d(out.size()[2])(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        return out
