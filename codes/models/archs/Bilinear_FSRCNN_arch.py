import functools  # 高阶函数工具
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F
import models.archs.arch_util as arch_util  # 自定义架构工具（用于权重初始化）
import torch


## 在ClassSR框架中，通过调整d（特征通道数）和m（映射层深度）实现简单、中等、复杂三级计算：
## 复杂度	d值	s值	m值	FLOPs量级	适用区域
## 简单	    16	12	2	~15M	平坦区域（天空）
## 中等	    36	12	3	~50M	中度纹理（植被）
## 复杂	    56	12	4	~120M	高频纹理（毛发）
## 不同复杂度区域的PSNR差距仅0.1~0.3dB，但计算量差异显著


# 功能 定义FSRCNN网络类
# 输入参数：
# input_channels：输入图像的通道数（如RGB图像为3）
# upscale：超分辨率缩放因子（如2x、4x）
# d：特征提取层的通道数（控制网络宽度）
# s：收缩层的通道数（降维程度）
# m：映射层的重复次数（控制网络深度）

# 输出参数：
# 返回一个FSRCNN网络实例，包含特征提取层、沙漏型主体结构和上采样层
# 该网络结构用于图像超分辨率重建任务，能够将低分辨率图像转换为高分辨率图像

class Bilinear_FSRCNN_net(torch.nn.Module):
    def __init__(self, input_channels, upscale, d=64, s=12, m=4):
        super(Bilinear_FSRCNN_net, self).__init__()
        # input_channels：输入通道数（如RGB为3）
        # upscale：超分辨率缩放因子（如2x、4x）

        # 关键参数：
        # d：特征提取层通道数（控制网络宽度）
        # s：收缩层通道数（降维程度）
        # m：映射层重复次数（控制网络深度）

        ## 功能：特征提取层
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU())
        # head_conv：特征提取层
        # in_channels = input_channels  # 输入通道数（如RGB图像为3）
        # out_channels=d  # 输出通道数（特征提取层的通道数）
        # kernel_size=5  # 卷积核大小为5x5
        # stride=1  # 步幅为1
        # padding=2  # 填充为2，保持输出尺寸与输入相同
        # PReLU激活函数增强非线性能力（相比ReLU避免梯度消失）

        ## 功能：沙漏型主体结构
        ## 设计亮点：通过“压缩-映射-扩展”结构，在保持性能的同时显著降低计算量

        self.layers = []  # 用于存储网络层

        # 收缩层：降维  目的是减少计算量和参数量
        # 将输入通道数从d压缩至s（s << d），减少计算量
        # 1x1卷积用于通道数压缩
        # in_channels=d：输入通道数
        # out_channels=s：输出通道数（收缩后的通道数）
        # kernel_size=1：卷积核大小为1x1
        # stride=1：步幅为1
        # padding=0：无填充
        # PReLU激活函数增强非线性能力
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        # 映射层：目的是提取特征
        # m 个 3x3卷积层，保持通道数为s，提取特征
        # 3x3卷积用于特征提取
        # in_channels=s：输入通道数（收缩后的通道数）
        # out_channels=s：输出通道数（保持不变）
        # kernel_size=3：卷积核大小为3x3
        # stride=1：步幅为1
        # padding=1：填充为1，保持输出尺寸与输入相同
        # PReLU激活函数增强非线性能力
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.PReLU())

        # 扩张层：恢复维度 目的是准备进行上采样
        # 将通道数从s恢复至d，准备进行上采样（s << d）
        # 1x1卷积用于通道数扩张
        # in_channels=s：输入通道数（映射后的通道数）
        # out_channels=d：输出通道数（恢复到特征提取层的通道数）
        # kernel_size=1：卷积核大小为1x1
        # stride=1：步幅为1
        # padding=0：无填充
        # PReLU激活函数增强非线性能力
        # 这里的s通常远小于d，因此通过1x1卷积将通道数从s恢复至d，准备进行上采样
        # 这样可以减少计算量，同时保持特征信息的完整性
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.body_conv = torch.nn.Sequential(*self.layers)
        # 收缩层（Shrinking） 1×1卷积将通道数从d压缩至s（s << d），减少计算量
        # 映射层（Mapping） m个3×3卷积层，保持通道数为s，提取特征
        # 扩张层（Expanding） 1×1卷积将通道数从s恢复至d，准备进行上采样

        ## 功能：反卷积层：上采样
        # 9×9转置卷积实现直接上采样（替代传统插值）
        # 将特征图从d通道上采样到input_channels通道（如RGB图像）
        # kernel_size=9：卷积核大小为9x9
        # stride=upscale：步幅为上采样因子（如2x、4x）
        # padding=3：填充为3，保持输出尺寸与输入相同
        # output_padding=1：输出填充为1，确保输出尺寸正确
        # in_channels=d：输入通道数（特征提取层的输出通道数）
        # out_channels=input_channels：输出通道数（如RGB图像的3个通道）
        # 反卷积层（Deconvolution）用于将特征图上采样到原始输入图像的尺寸

        # Deconvolution
        # self.tail_conv = nn.ConvTranspose2d(in_channels=d, out_channels=input_channels, kernel_size=9,
        #                                     stride=upscale, padding=3, output_padding=1)

        # ===== 原版：双线性上采样 + 3x3卷积微调（两层） =====
        self.tail_conv = nn.Sequential(
            # 双线性上采样
            nn.Upsample(scale_factor=upscale, mode='bilinear', align_corners=False),

            # 3x3 卷积微调（两层）
            nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1),
            nn.PReLU(d),
            nn.Conv2d(d, out_channels=input_channels, kernel_size=3, stride=1, padding=1)
        )

        ## 功能：权重初始化
        # 使用正态分布初始化卷积层权重，标准差为0.1，控制初始化范围，避免梯度爆炸
        # 使用自定义工具初始化权重（如Xavier或He初始化）
        arch_util.initialize_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)

    ## 功能：前向传播流程
    # 输入：x - 输入图像张量，形状为(batch_size, input_channels, height, width)
    # 输出：out - 超分辨率图像张量，形状为(batch_size, input_channels, height * upscale, width * upscale)
    # 前向传播流程：特征提取 → 沙漏型主体 → 上采样
    # 特征提取层提取输入图像特征，沙漏型主体进行特征变换，上采样层将特征图上采样到原始输入图像尺寸
    def forward(self, x):
        fea = self.head_conv(x)  # 特征提取层
        fea = self.body_conv(fea)  # 沙漏型主体结构
        out = self.tail_conv(fea)  # 上采样层
        return out
