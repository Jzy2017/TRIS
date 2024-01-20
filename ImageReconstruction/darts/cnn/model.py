import torch
import torch.nn as nn
from darts.cnn.operations import *
from torch.autograd import Variable
from darts.cnn.utils import drop_path
import darts.cnn.genotypes
import numpy as np
import torch.nn.functional as F


l1loss=nn.L1Loss()
l2loss=nn.MSELoss()

class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    #print(C_prev_prev, C_prev, C)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkCIFAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux

def edge_extraction(gen_frames, use_cuda):
    # calculate the loss for each scale
    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.

    channels = gen_frames.shape[1]
    pos = torch.from_numpy(np.identity(channels))  # 3 x 3
    neg = -1 * pos
    filter_x = torch.cat([neg, pos], 0).unsqueeze(0)
    filter_y = torch.cat([neg, pos], 1).unsqueeze(0)
    # print(neg.shape)

    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'
    conv_x = nn.Conv2d(1, 1, kernel_size=(2, 1), stride=1, bias=False, padding=0)
    # conv_x = nn.Conv2d(1,1,3,bias=False)
    conv_y = nn.Conv2d(1, 1, kernel_size=(1, 2), stride=1, bias=False, padding=0)
    conv_x.requires_grad = False
    conv_y.requires_grad = False
    x_pad = nn.ZeroPad2d((0, 0, 1, 0))
    y_pad = nn.ZeroPad2d((1, 0, 0, 0))
    if use_cuda:
        filter_x = filter_x.cuda()
        filter_y = filter_y.cuda()
        conv_x = conv_x.cuda()
        conv_y = conv_y.cuda()
        x_pad = x_pad.cuda()
        y_pad = y_pad.cuda()
    conv_x.weight.data = filter_x.unsqueeze(0).float()
    conv_y.weight.data = filter_y.unsqueeze(0).float()

    # a=conv_x(gen_frames)
    gen_dx = torch.abs(conv_x(x_pad(gen_frames)))
    gen_dy = torch.abs(conv_y(y_pad(gen_frames)))

    edge = gen_dx ** 1 + gen_dy ** 1
    edge_clip = torch.clamp(edge, 0, 1)
    # condense into one tensor and avg
    return edge_clip


def seammask_extraction(mask, use_cuda):
    seam_mask = edge_extraction(torch.unsqueeze(torch.mean(mask, axis=1), 1), use_cuda)
    filters = torch.from_numpy(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])).unsqueeze(0).unsqueeze(
        0).float()
    conv_1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False, padding=1)
    conv_2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False, padding=1)
    conv_3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False, padding=1)
    conv_2.requires_grad = False
    conv_2.requires_grad = False
    conv_3.requires_grad = False

    if use_cuda:
        conv_1 = conv_1.cuda()
        conv_2 = conv_2.cuda()
        conv_3 = conv_3.cuda()
        filters = filters.cuda()
    conv_1.weight.data = filters
    conv_2.weight.data = filters
    conv_3.weight.data = filters
    test_conv1 = conv_1(seam_mask)
    test_conv1 = torch.clamp(test_conv1, 0, 1)
    test_conv2 = conv_2(test_conv1)
    test_conv2 = torch.clamp(test_conv2, 0, 1)
    test_conv3 = conv_3(test_conv2)
    test_conv3 = torch.clamp(test_conv3, 0, 1)
    # condense into one tensor and avg
    return test_conv3


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Restructor(nn.Module):
    def __init__(self, n_channels, n_classes, genotype, bilinear=True):
        super(Restructor, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        stem_multiplier = 3
        C_curr = stem_multiplier * n_channels
        self.stem = nn.Sequential(
            nn.Conv2d(6, C_curr, 3, padding=1, bias=False)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, 16
        self.cells = nn.ModuleList()
        reduction_prev = False
        layers=1
        for i in range(layers):
            reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.CB512 = ContextBlock(inplanes=512, ratio=1)
        self.CB256 = ContextBlock(inplanes=256, ratio=1)
        self.CB128 = ContextBlock(inplanes=128, ratio=2)
        self.CB64 = ContextBlock(inplanes=64, ratio=3)

    def forward(self, x):

        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, x1 = s0, cell(s0, s1, 0.1)
        # print(x1.shape)
        x2 = self.down1(x1)  # 1, 128, 128, 192
        # print(x2.shape)
        x3 = self.down2(x2)  # 1, 256, 64, 96
        # print(x3.shape)
        x4 = self.down3(x3)  # 1, 512, 32, 48
        # print(x4.shape)
        x5 = self.down4(x4)  # 1, 512, 16, 24
        # print(x5.shape)
        x = self.up1(x5, self.CB512(x4))  # 1, 256, 32, 48
        # print(x.shape)
        # print(x.shape)
        # print(x3.shape)
        x = self.up2(x, self.CB256(x3))  # 1, 128, 64, 96
        # print(x.shape)
        # print(x.shape)
        # time.sleep(100)
        x = self.up3(x, self.CB128(x2))  # 1, 64, 128, 192
        # print(x.shape)
        x = self.up4(x, self.CB64(x1))  # 1, 64, 256, 384
        # print(x.shape)
        # x = self.up4(x, x1)
        logits = self.outc(x)
        # time.sleep(1000)
        return logits

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        nn.init.constant_(m[-1].weight, val=0)
    else:
        nn.init.constant_(m.weight, val=0)


class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            nn.init.kaiming_normal_(self.conv_mask.weight, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:# init False
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:# init True
            # [N, C, 1, 1]
            # print("haha")
            channel_add_term = self.channel_add_conv(context)
            # print(channel_add_term.shape)
            # print(out.shape)
            out = out + channel_add_term
            # print(out.shape)

        return out



if __name__ == '__main__':
    genotype = eval("genotypes.%s" % 'DARTS')
    #print(genotype)
    net = NetworkBackbone(C=16, num_classes=10, layers=4, auxiliary=False, genotype=genotype).cuda()
    net.drop_path_prob = 0.1
    height = 128
    width = 128

    x = torch.randn((4, 3, height, width)).cuda()
    y1, y2, y3, y4 = net(x)
    print(y1.shape, y2.shape, y3.shape, y4.shape)
    #print(net)
