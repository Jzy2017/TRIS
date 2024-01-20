import torch
import torch.nn as nn
import torch.nn.functional as F
from darts.cnn.operations import *
from torch.autograd import Variable
from darts.cnn.genotypes import PRIMITIVES
from darts.cnn.genotypes import Genotype
import numpy as np
#from stitch.output_tensorDLT import output_solve_DLT
from output_tensorDLT import output_solve_DLT
#from stitch.attention import Attention1D
from attention import Attention1D
#from stitch.position import PositionEmbeddingSine
from position import PositionEmbeddingSine
#from stitch.tensorDLT_function import solve_DLT
from tensorDLT_function import solve_DLT
#from stitch.spatial_transform import Transform
from spatial_transform import Transform
#from stitch.output_spatial_transform import Transform_output
from output_spatial_transform import Transform_output

l1loss=nn.L1Loss()
l2loss=nn.MSELoss()

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

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
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


def cost_volume(c1, warp, search_range,pad,lrelu):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = pad(warp)
    _,c ,h, w = c1.shape
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice=padded_lvl[:, :,y:y+h, x:x+w]
            cost = torch.mean(c1 * slice, axis=1, keepdims=True)
            cost_vol.append(cost)
    cost_vol = torch.cat(cost_vol, axis=1)
    cost_vol = lrelu(cost_vol)
    return cost_vol
def vertical_cost_volume(c1, warp, search_range,pad,lrelu):
    """Build vertical cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = pad(warp)
    _,c ,h, w = c1.shape
    max_offset = search_range * 2 + 1
    cost_vol = []
    for y in range(0, max_offset):
        slice=padded_lvl[:, :,y:y+h, :]
        cost = torch.mean(c1 * slice, axis=1, keepdims=True)
        cost_vol.append(cost)
    cost_vol = torch.cat(cost_vol, axis=1)
    cost_vol = lrelu(cost_vol)
    return cost_vol
def horizontal_cost_volume(c1, warp, search_range,pad,lrelu):
    """Build horizontal cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """

    padded_lvl = pad(warp)
    _,c ,h, w = c1.shape
    max_offset = search_range * 2 + 1
    cost_vol = []
    for x in range(0, max_offset):
        slice=padded_lvl[:, :,:, x:x+w]
        cost = torch.mean(c1 * slice, axis=1, keepdims=True)
        cost_vol.append(cost)
    cost_vol = torch.cat(cost_vol, axis=1)
    cost_vol = lrelu(cost_vol)
    return cost_vol

class RegressionNet(torch.nn.Module):
  def __init__(self, is_training, search_range, warped=False):
    super().__init__()
    self.search_range = search_range
    self.warped = warped
    self.keep_prob = 0.5 if is_training == True else 1.0
    self.feature = []
    if self.search_range == 16:
      self.in_channel = 128
      self.stride = [1, 1, 1]
      self.out = [128, 128, 128]
      self.fully = 1024
    elif self.search_range == 8:
      self.in_channel = 128
      self.stride = [1, 1, 2]
      self.out = [128, 128, 128]
      self.fully = 512
    elif self.search_range == 4:
      self.in_channel = 64
      self.stride = [1, 2, 2]
      self.out = [128, 128, 128]
      self.fully = 256

    # self.pad=nn.ConstantPad2d(value = 0,padding = [search_range, search_range, search_range, search_range])
    self.vertical_pad = nn.ConstantPad2d(value=0, padding=[0, 0, search_range, search_range])
    self.horizontal_pad = nn.ConstantPad2d(value=0, padding=[search_range, search_range, 0, 0])
    # self.pad=nn.ConstantPad2d(value = 0,padding = [search_range, search_range, search_range, search_range])
    self.lrelu = nn.LeakyReLU(inplace=True)
    # self.conv1 =torch.nn.Sequential( nn.Conv2d(in_channels=((2*self.search_range)+1)*2, out_channels=self.out[0], kernel_size=3, stride=self.stride[0],padding=1),
    #                                 nn.BatchNorm2d(self.out[0]),
    #                                 nn.ReLU(True))
    self.conv1 = torch.nn.Sequential(
      nn.Conv2d(in_channels=((2 * self.search_range) + 1) * 2, out_channels=self.out[0], kernel_size=3,
                stride=self.stride[0], padding=1),
      nn.BatchNorm2d(self.out[0]),
      nn.ReLU(True))
    self.conv2 = torch.nn.Sequential(
      nn.Conv2d(in_channels=self.out[0], out_channels=self.out[1], kernel_size=3, stride=self.stride[1], padding=1),
      nn.BatchNorm2d(self.out[1]),
      nn.ReLU(True))
    self.conv3 = torch.nn.Sequential(
      nn.Conv2d(in_channels=self.out[1], out_channels=self.out[2], kernel_size=3, stride=self.stride[2], padding=1),
      nn.BatchNorm2d(self.out[2]),
      nn.ReLU(True))
    self.getoffset = torch.nn.Sequential(torch.nn.Linear(in_features=256 * self.out[2], out_features=self.fully),
                                         nn.ReLU(True),
                                         nn.Dropout(p=self.keep_prob),
                                         torch.nn.Linear(in_features=self.fully, out_features=8))
    self.horizontal_attention = Attention1D(self.in_channel,
                                            y_attention=False,
                                            double_cross_attn=True,
                                            )
    self.vertical_attention = Attention1D(self.in_channel,
                                          y_attention=True,
                                          double_cross_attn=True,
                                          )

  def forward(self, feature1, feature2):
    # print(feature1.shape)
    pos_channels = feature1.shape[1] // 2
    pos_enc = PositionEmbeddingSine(pos_channels)
    position = pos_enc(feature1)  # [B, C, H, W]
    if not self.warped:
      verti_attn, attn_y = self.vertical_attention(nn.functional.normalize(feature1, dim=1, p=2),
                                                   nn.functional.normalize(feature2, dim=1, p=2), position)
      hori_attn, attn_x = self.horizontal_attention(nn.functional.normalize(feature1, dim=1, p=2),
                                                    nn.functional.normalize(feature2, dim=1, p=2), position)
      # correlation = cost_volume(nn.functional.normalize(feature1, dim=1, p=2), nn.functional.normalize(feature2,dim=1, p=2), self.search_range,self.pad,self.lrelu)
    else:
      verti_attn, attn_y = self.vertical_attention(nn.functional.normalize(feature1, dim=1, p=2), feature2, position)
      hori_attn, attn_x = self.horizontal_attention(nn.functional.normalize(feature1, dim=1, p=2), feature2, position)
      # correlation = cost_volume(nn.functional.normalize(feature1, dim=1, p=2), feature2, self.search_range,self.pad,self.lrelu)
    verti_correlation = vertical_cost_volume(nn.functional.normalize(feature1, dim=1, p=2), hori_attn,
                                             self.search_range, self.vertical_pad, self.lrelu)
    hori_correlation = horizontal_cost_volume(nn.functional.normalize(feature1, dim=1, p=2), verti_attn,
                                              self.search_range, self.horizontal_pad, self.lrelu)
    # verti_correlation = vertical_cost_volume(nn.functional.normalize(feature1, dim=1, p=2), hori_attn, self.search_range,self.vertical_pad,self.lrelu)
    # correlation = cost_volume(nn.functional.normalize(feature1, dim=1, p=2),nn.functional.normalize(feature2,dim=1, p=2), self.search_range, self.pad, self.lrelu)
    correlation = torch.cat((hori_correlation, verti_correlation), 1)
    conv1 = self.conv1(correlation)
    conv2 = self.conv2(conv1)
    conv3 = self.conv3(conv2)
    conv3_flatten = conv3.contiguous().view(conv3.shape[0], -1)
    offset = self.getoffset(conv3_flatten)
    return offset


class NetworkStitch(nn.Module):

  def __init__(self, C=16, layers=4, steps=4, multiplier=4, stem_multiplier=3, is_training=1, batch_size=1):
    super(NetworkStitch, self).__init__()
    self._C = C
    self.batch_size = batch_size
    self._layers = layers
    self._steps = steps
    self._multiplier = multiplier
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # two feature extractor with DARTS
    self.maxpool_vis = []
    for i in range(4):
      self.maxpool_vis.append(torch.nn.MaxPool2d((2, 2), stride=2))

    C_curr_vis = stem_multiplier * C
    self.stem_vis = nn.Sequential(
      nn.Conv2d(3, C_curr_vis, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr_vis)
    )

    C_prev_prev_vis, C_prev_vis, C_curr_vis = C_curr_vis, C_curr_vis, C
    self.cells_vis = nn.ModuleList()
    reduction_prev_vis = False

    for i in range(layers):
      if i in [2]:
        C_curr_vis *= 2
        reduction_vis = True
      else:
        reduction_vis = False

      cell_vis = Cell(steps, multiplier, C_prev_prev_vis, C_prev_vis, C_curr_vis, reduction_vis, reduction_prev_vis)

      reduction_prev_vis = reduction_vis

      self.cells_vis += [cell_vis]

      C_prev_prev_vis, C_prev_vis = C_prev_vis, multiplier * C_curr_vis

    self._initialize_alphas()

    # other parts of stitching model of H_estimater
    self.Rnet1 = RegressionNet(is_training, 16, warped=False)  # .cuda()
    self.Rnet2 = RegressionNet(is_training, 8, warped=True)  # .cuda()
    self.Rnet3 = RegressionNet(is_training, 4, warped=True)  # .cuda()
    for m in self.Rnet1.modules():
      if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
    for m in self.Rnet2.modules():
      if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
    for m in self.Rnet3.modules():
      if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)

    self.M_tile_inv_128, self.M_tile_128 = self.to_transform_H(128, batch_size)
    self.M_tile_inv_32, self.M_tile_32 = self.to_transform_H(32, batch_size)
    self.M_tile_inv_64, self.M_tile_64 = self.to_transform_H(64, batch_size)
    # self.solve_DLT = DLT(batch_size).to(self.device)
    # self.output_solve_DLT = output_DLT(batch_size).to(self.device)
    self.transform32 = Transform(32, 32, self.device, batch_size).to(self.device)
    self.transform64 = Transform(64, 64, self.device, batch_size).to(self.device)
    self.transform128 = Transform(128, 128, self.device, batch_size).to(self.device)


  def new(self):
    model_new = NetworkStitch(batch_size=self.batch_size).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
      x.data.copy_(y.data)
    return model_new

  def to_transform_H(self, patch_size, batch_size):
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = torch.from_numpy(M)
    M_tile = torch.unsqueeze(M_tensor, 0).repeat([batch_size, 1, 1])
    M_inv = np.linalg.inv(M)
    M_tensor_inv = torch.from_numpy(M_inv)
    M_tile_inv = torch.unsqueeze(M_tensor_inv, 0).repeat([batch_size, 1, 1])
    M_tile_inv = M_tile_inv.to(self.device)
    M_tile = M_tile.to(self.device)
    return M_tile_inv, M_tile

  def vis_feature_extract(self, input):
    # extract ir features
    y = []
    s0 = s1 = self.stem_vis(input)
    for i, cell in enumerate(self.cells_vis):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
      y.append(s1)
      if i ==0 or i ==2:
        s0 = self.maxpool_vis[i](s0)
        s1 = self.maxpool_vis[i](s1)

    return y[0], y[1], y[2], y[3]

  def forward(self, inputs, is_test=False, patch_size=128., size=None):

    vis_input1 = inputs[..., 0:3].permute(0, 3, 1, 2)
    vis_input2 = inputs[..., 3:6].permute(0, 3, 1, 2)
    vis_input1 = torch.nn.functional.interpolate(vis_input1, [128, 128])
    vis_input2 = torch.nn.functional.interpolate(vis_input2, [128, 128])

    # feature extractor
    vis_feature1 = self.vis_feature_extract(vis_input1.to(torch.float32))
    vis_feature2 = self.vis_feature_extract(vis_input2.to(torch.float32))
    #return ir_feature1
    ################################ vis #####################################################
    ##############################  Regression Net 1 ##############################
    vis_net1_f = self.Rnet1(vis_feature1[-1], vis_feature2[-1])
    vis_net1_f = torch.unsqueeze(vis_net1_f, 2)  # *128
    # H1 = self.DLT(net1_f/4., 32.)
    vis_H1 = solve_DLT(vis_net1_f / 4., 32.)
    vis_H1 = torch.matmul(torch.matmul(self.M_tile_inv_32, vis_H1), self.M_tile_32)
    vis_feature2_warp = self.transform32(nn.functional.normalize(vis_feature2[-2], dim=1, p=2), vis_H1)
    # ##############################  Regression Net 2 ##############################
    vis_net2_f = self.Rnet2(vis_feature1[-2], vis_feature2_warp)
    vis_net2_f = torch.unsqueeze(vis_net2_f, 2)  # *128
    # H2 = self.DLT((net1_f+net2_f)/2., 64.)
    vis_H2 = solve_DLT((vis_net1_f + vis_net2_f) / 2., 64.)
    vis_H2 = torch.matmul(torch.matmul(self.M_tile_inv_64, vis_H2), self.M_tile_64)
    vis_feature3_warp = self.transform64(nn.functional.normalize(vis_feature2[-3], dim=1, p=2), vis_H2)
    # ##############################  Regression Net 3 ##############################
    vis_net3_f = self.Rnet3(vis_feature1[-3], vis_feature3_warp)
    vis_net3_f = torch.unsqueeze(vis_net3_f, 2)  # *128
    if is_test:
      return vis_net1_f + vis_net2_f + vis_net3_f
    ###############################################################################

    vis_H1 = solve_DLT(vis_net1_f, 128)
    vis_H2 = solve_DLT(vis_net1_f + vis_net2_f, 128)
    vis_H3 = solve_DLT(vis_net1_f + vis_net2_f + vis_net3_f, 128)
    vis_H1_mat = torch.matmul(torch.matmul(self.M_tile_inv_128, vis_H1), self.M_tile_128)
    vis_H2_mat = torch.matmul(torch.matmul(self.M_tile_inv_128, vis_H2), self.M_tile_128)
    vis_H3_mat = torch.matmul(torch.matmul(self.M_tile_inv_128, vis_H3), self.M_tile_128)
    vis_image2_tensor = vis_input2
    vis_image1_tensor = vis_input1
    vis_warp2_H1 = self.transform128(vis_image2_tensor, vis_H1_mat)
    vis_warp2_H2 = self.transform128(vis_image2_tensor, vis_H2_mat)
    vis_warp2_H3 = self.transform128(vis_image2_tensor, vis_H3_mat)
    one = torch.ones_like(vis_image2_tensor)
    vis_warp1_H1 = self.transform128(one, vis_H1_mat) * vis_image1_tensor
    vis_warp1_H2 = self.transform128(one, vis_H2_mat) * vis_image1_tensor
    vis_warp1_H3 = self.transform128(one, vis_H3_mat) * vis_image1_tensor
    vis_warp2 = torch.cat((vis_warp2_H1, vis_warp2_H2, vis_warp2_H3), 1)
    vis_warp1 = torch.cat((vis_warp1_H1, vis_warp1_H2, vis_warp1_H3), 1)

    return vis_net1_f, vis_net2_f, vis_net3_f, vis_warp1, vis_warp2


  def _loss(self, input):
    vis_off1, vis_off2, vis_off3, vis_warp1, vis_warp2= self(input)
    vis_l1_1 = 16 * l1loss(vis_warp1[:, 0:3, ...], vis_warp2[:, 0:3, ...])
    vis_l1_2 = 4 * l1loss(vis_warp1[:, 3:6, ...], vis_warp2[:, 3:6, ...])
    vis_l1_3 = 2 * l1loss(vis_warp1[:, 6:9, ...], vis_warp2[:, 6:9, ...])
    loss_unsupervise = vis_l1_1 + vis_l1_2 + vis_l1_3
    loss_all = loss_unsupervise
    return loss_all

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2 + i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2),
                       key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2 + self._steps - self._multiplier, self._steps + 2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

if __name__ == '__main__':
    #genotype = eval("genotypes.%s" % 'DARTS')
    #print(genotype)
    gt = torch.randn((1,8,1)).cuda()
    net = NetworkStitch(batch_size=1).cuda()
    #net.drop_path_prob = 0.1
    height = 128
    width = 128

    x = torch.randn((1, height, width, 12)).cuda()
    y = net(x, gt=gt)
    for yi in y:
      print(yi.shape)

