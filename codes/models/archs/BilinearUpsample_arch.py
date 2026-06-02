import functools  # 高阶函数工具
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F
import models.archs.arch_util as arch_util  # 自定义架构工具（用于权重初始化）
import torch

# ==================== 修改点1：定义双线性上采样模块 ====================
class BilinearUpsample_net(nn.Module):
    """轻量级双线性上采样模块，带卷积微调"""

    def __init__(self, upscale=4):
        super(BilinearUpsample_net, self).__init__()
        self.scale = upscale

        # 微调卷积层 - 轻量级后处理
        self.post_conv = nn.Sequential(
             nn.Conv2d(3, 8, 3, padding=1),  # 3x3卷积提取特征
             nn.PReLU(),  # 参数化ReLU增强非线性
             nn.Conv2d(8, 3, 1)  # 1x1卷积融合特征
        )

    def forward(self, x):
        # 双线性上采样
        x_up = F.interpolate(
            x,
            scale_factor=self.scale,
            mode='bilinear',
            align_corners=False
        )

        # # 微调卷积处理
        return self.post_conv(x_up)
