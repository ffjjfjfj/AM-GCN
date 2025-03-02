import torch
import torch.nn as nn

# from .utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=groups)


# CA BLOCK
class CABlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        # 接收输入平面数 (inplanes)，输出平面数 (planes)，步长 (stride)，下采样 (downsample)，组数 (groups)，
        # 基础宽度 (base_width)，扩张度 (dilation) 和规范层 (norm_layer) 作为输入
        super(CABlock, self).__init__()
        # 如果未指定规范层，则默认使用 nn.BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.bn1 = norm_layer(planes)  # 批量归一化层
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes, groups=groups)
        self.bn2 = norm_layer(planes)  # 批量归一化层
        self.attn = nn.Sequential(  # 注意力机制的序列
            nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=False),  # 32*33*33
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.downsample = downsample
        self.stride = stride
        self.planes = planes

    def forward(self, x):
        x, attn_last, if_attn = x  ##attn_last: downsampled attention maps from last layer as a prior knowledge
        identity = x    # 24*256*28*28

        out = self.conv1(x)   # 24*128*112*112/24*128*56*56/24*256*28*28
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:     # 如果存在下采样操作
            identity = self.downsample(identity)    # 则对identity进行下采样操作

        out = self.relu(out + identity)     # 将第二次归一化后的数据与identity相加，并经过ReLU激活函数得到残差块的输出out
        # 对out沿着通道维度计算均值avg_out和最大值max_out，然后将它们拼接在一起形成注意力图attn
        avg_out = torch.mean(out, dim=1, keepdim=True)      # 24*1*112*112/24*1*56*56/24*1*28*28/24*1*14*14
        max_out, _ = torch.max(out, dim=1, keepdim=True)    # 24*1*112*112
        attn = torch.cat((avg_out, max_out), dim=1)
        # 将注意力图attn通过注意力机制self.attn得到注意力权重。
        attn = self.attn(attn)
        # 如果上一层的注意力图attn_last不为None，则将当前注意力图乘以上一层的注意力图
        if attn_last is not None:
            attn = attn_last * attn

        attn = attn.repeat(1, self.planes, 1, 1)
        if if_attn:
            out = out * attn

        # 三个输出与“x, attn_last, if_attn = x”三个量相对应
        return out, attn[:, 0, :, :].unsqueeze(1), True


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=4, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(90 * 2, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False, groups=1)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 128, layers[0], groups=1)
        self.inplanes = int(self.inplanes * 1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], groups=1)
        self.inplanes = int(self.inplanes * 1)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], groups=1)
        self.inplanes = int(self.inplanes * 1)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], groups=1)
        self.inplanes = int(self.inplanes * 1)

        self.fc = nn.Linear(512 * block.expansion * 196, 5)     # 几分类
        self.drop = nn.Dropout(p=0.1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, groups=1):
        norm_layer = self._norm_layer  # 归一化
        downsample = None
        previous_dilation = self.dilation
        if dilate:  # 扩张卷积的参数
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, POS):  ##x->input of main branch; POS->position embeddings generated by sub branch

        x = self.conv1(x)  # 卷积     24*128*112*112
        x = self.bn1(x)  # 归一化
        x = self.relu(x)  # 激活函数
        # main branch   主分支
        x, attn1, _ = self.layer1((x, None, True))  # 自注意力机制
        temp = attn1
        attn1 = self.maxpool(attn1)  # 最大池化     24*1*56*56

        x, attn2, _ = self.layer2((x, attn1, True))  # 使用第一次注意力作为输入

        attn2 = self.maxpool(attn2)     # 24*1*28*28

        x, attn3, _ = self.layer3((x, attn2, True))
        #
        attn3 = self.maxpool(attn3)     # 24*1*14*14
        x, attn4, _ = self.layer4((x, attn3, True))     # 24*1*14*14

        # fusion of motion pattern feature and position embeddings    运动模式特征与位置嵌入的融合
        # x:24*512*14*14    pos:24*512*14*14
        # x = x + POS

        x = torch.flatten(x, 1)  # 对融合后的结果进行平铺操作    24*100352

        x = self.fc(x)  # 传入到全连接层   24*5

        return x, temp.view(x.size(0), -1)

    def forward(self, x, POS):
        return self._forward_impl(x, POS)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


##main branch consisting of CA blocks
def resnet18_pos_attention(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', CABlock, [1, 1, 1, 1], pretrained, progress,
                   **kwargs)
