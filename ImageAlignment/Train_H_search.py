import torch

import os
from darts.cnn.architect import Architect
import torch.nn.functional as F

from models import disjoint_augment_image_pair  # ,H_estimator
# from H_model_mini import H_estimator
from darts.cnn.model_search import NetworkStitch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled=True
# from H_model_detone import H_estimator
import torch.nn as nn
import numpy as np
import argparse
import glob
import sys
import cv2
import random
from align_attacks.evaluate_attacks_pgb import _fgsm_whitebox, _pgd_whitebox, _bim_whitebox
from darts.cnn import utils
from dataset import Image_stitch, Image_UDIS
import time
import logging
import time
from torch.autograd import Variable

# viz = Visdom(env='ZZX-stitch-pytorch')  # 启用可视化工具
# viz.line([0.], [0], win='total_loss', opts = dict(title='total_loss'))
# viz.line([0.], [0], win='l1_1_loss', opts = dict(title='l1_1_loss'))
# viz.line([0.], [0], win='l1_2_loss', opts = dict(title='l1_2_loss'))
# viz.line([0.], [0], win='l1_3_loss', opts = dict(title='l1_3_loss'))

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def environment_check():
    if torch.cuda.is_available():
        # os.system('gpustat')
        i = int(input("choose devise:"))

        if i != -1:
            torch.cuda.set_device(device=i)
            return i
    print("cuda: False")
    return 'cpu'


device = environment_check()
if device != 'cpu':
    use_cuda = True
else:
    use_cuda = False


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=1e-7, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=1e-2, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-4, help='weight decay for arch encoding')
args = parser.parse_args()


args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

vis_batch = 50  # for checking loss during training
mode = 'unsupervise'  # optinal in ['supervise','unsupervise']
dataset_mode = 'mix'  # optinal in ['coco','roadscene','mix']
#num_epochs = 100
#learning_rate = 0.0002
height, width = 128, 128
#batch_size = 8
data_root='../data/UDIS-D/training/'
# netR_path='snapshot/32_R.pkl'
# netH_path='snapshot/32_H.pkl'
netR_path = None
data = Image_UDIS(vis1_path=os.path.join(data_root, 'input1'), \
                    vis2_path=os.path.join(data_root, 'input2'))

num_train = len(data)
indices = list(range(num_train))
split = int(np.floor(args.train_portion * num_train))


train_queue = torch.utils.data.DataLoader(
  data, batch_size=args.batch_size,
  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
  pin_memory=True, num_workers=0)

valid_queue = torch.utils.data.DataLoader(
  data, batch_size=args.batch_size,
  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
  pin_memory=True, num_workers=0)

l1loss = nn.L1Loss()
l2loss = nn.MSELoss()

# netR = H_estimator(batch_size=batch_size,device=device,is_training=1)
netR = NetworkStitch(batch_size=args.batch_size)
if netR_path is not None:
    netR.load_state_dict(torch.load(netR_path, map_location='cpu'))
if use_cuda:
    l1loss = l1loss.to(device)
    l2loss = l2loss.to(device)
    netR = netR.to(device)
optimizerR = torch.optim.Adam(netR.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizerR, float(args.epochs), eta_min=args.learning_rate_min)

architect = Architect(netR, args)

save_folder = 'snapshot_search'
# define dataset
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
loss_all_batch = 0
if mode == 'unsupervise':
    l1_1_batch = 0
    l1_2_batch = 0
    l1_3_batch = 0
    l1_out_batch = 0
elif mode == 'supervise':
    l2_gt_1_batch = 0
    l2_gt_2_batch = 0
    l2_gt_3_batch = 0
    l2_gt_out_batch = 0
netR.train()
# netR.eval()
for epoch in range(0, args.epochs + 1):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = netR.genotype()
    logging.info('genotype = %s', genotype)

    a1 = F.softmax(netR.alphas_normal, dim=-1)
    a2 = F.softmax(netR.alphas_reduce, dim=-1)

    logging.info('model.alphas_normal = \n {}'.format(a1))
    logging.info('model.alphas_reduce = \n {}'.format(a2))

    for i, (vis_input1, vis_input2, size, name) in enumerate(train_queue):

        # train_ir_inputs_aug = disjoint_augment_image_pair(ir_input1,ir_input2)
        train_vis_inputs_aug = disjoint_augment_image_pair(vis_input1, vis_input2)
        if use_cuda:
            # vis_input1=vis_input1.to(device)
            # vis_input2=vis_input2.to(device)
            train_vis_inputs_aug = train_vis_inputs_aug.to(device)
            size = size.to(device)

        # train_vis_inputs = torch.cat((vis_input1,vis_input2), 3)
        # 生成对抗样本

        random_number = random.random()
        if random_number > 0.5:
            train_vis_inputs_aug = _pgd_whitebox(netR, train_vis_inputs_aug)

        vis_input1_search, vis_input2_search, size_search, \
        name_search = next(iter(valid_queue))
        input_search = torch.cat((vis_input1_search, vis_input2_search), 3)
        input_search = Variable(input_search, requires_grad=False).cuda()
        random_number = random.random()
        if random_number > 0.5:
            input_search = _pgd_whitebox(netR, input_search)

        architect.step(train_vis_inputs_aug, input_search, lr, optimizerR, unrolled=args.unrolled)
        optimizerR.zero_grad()

        vis_off1, vis_off2, vis_off3, vis_warp1, vis_warp2 = netR(train_vis_inputs_aug)
        # vis_off1, vis_off2, vis_off3, vis_warp2_H1, vis_warp2_H2, vis_warp2_H3, vis_one_warp_H1, vis_one_warp_H2, vis_one_warp_H3 , vis_warp2_gt, vis_one_warp_gt= netR(vis1_feature, vis2_feature,train_vis_inputs,gt)
        vis_shift = vis_off1 + vis_off2 + vis_off3
        ##### unsupervise training  (optional)##################################################################################
        if mode == 'unsupervise':
            vis_l1_1 = 16 * l1loss(vis_warp1[:, 0:3, ...], vis_warp2[:, 0:3, ...])
            vis_l1_2 = 4 * l1loss(vis_warp1[:, 3:6, ...], vis_warp2[:, 3:6, ...])
            vis_l1_3 = 2 * l1loss(vis_warp1[:, 6:9, ...], vis_warp2[:, 6:9, ...])
            loss_unsupervise = vis_l1_1 + vis_l1_2 + vis_l1_3
            loss_all = loss_unsupervise
            l1_1_batch += (vis_l1_1.item())
            l1_2_batch += (vis_l1_2.item())
            l1_3_batch += (vis_l1_3.item())

        loss_all_batch += loss_all.item()

        if i % vis_batch == 0 and i != 0:
            # print('shift: '+str(train_net1_f[0].reshape((1,8)).detach().cpu().numpy()))
            if mode == 'unsupervise':
                print(
                    '[%d/%d][%d/%d]\tlr_rate: %0.4f \tLoss_total: %.4f\t Loss_1: %.4f\t Loss_2: %.4f\t Loss_3: %.4f\t' %
                    (epoch, args.epochs, i, len(train_queue), optimizerR.state_dict()['param_groups'][0]['lr'],
                     loss_all_batch / vis_batch, l1_1_batch / vis_batch, l1_2_batch / vis_batch,
                     l1_3_batch / vis_batch))
                l1_1_batch = 0
                l1_2_batch = 0
                l1_3_batch = 0
            elif mode == 'supervise':
                print(
                    '[%d/%d][%d/%d]\tlr_rate: %0.4f \tLoss_total: %.4f\t Loss_1: %.4f\t Loss_2: %.4f\t Loss_3: %.4f\t' %
                    (epoch, args.epochs, i, len(train_queue), optimizerR.state_dict()['param_groups'][0]['lr'],
                     loss_all_batch / vis_batch, l2_gt_1_batch / vis_batch, l2_gt_2_batch / vis_batch,
                     l2_gt_3_batch / vis_batch))
                l2_gt_1_batch = 0
                l2_gt_2_batch = 0
                l2_gt_3_batch = 0
            loss_all_batch = 0

        if i % vis_batch == 0:
            vis_warp1s = torch.cat((train_vis_inputs_aug[0][..., 0:3].permute(2, 0, 1), vis_warp1[0, 0:3],
                                    vis_warp1[0, 3:6], vis_warp1[0, 6:9]), 1)
            vis_warp2s = torch.cat((train_vis_inputs_aug[0][..., 3:6].permute(2, 0, 1), vis_warp2[0, 0:3],
                                    vis_warp2[0, 3:6], vis_warp2[0, 6:9]), 1)
            visualize = torch.cat((vis_warp1s, vis_warp2s), 2).permute(1, 2, 0).detach().cpu().numpy() * 255
            cv2.imwrite('visual.png', visualize)
        loss_all.backward()
        optimizerR.step()
        # optimizerH.step()
        #optimizerR.zero_grad()
        # optimizerH.zero_grad()
    if epoch % 4 == 0:
        torch.save(netR.state_dict(), save_folder + '/' + str(epoch) + "_search.pkl")
        # torch.save(net_vis.state_dict(), save_folder+ '/' + str(epoch) + "_V.pkl")
        # torch.save(net_ir.state_dict(), save_folder+ '/' + str(epoch) + "_I.pkl")
        # torch.save(netH.state_dict(), save_folder+ '/' + str(epoch) + "_H.pkl")

