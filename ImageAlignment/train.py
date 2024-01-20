import os
import torch
from models import disjoint_augment_image_pair  # ,H_estimator
# from H_model_mini import H_estimator
from H_model import H_joint, H_estimator
import torch.backends.cudnn as cudnn
from darts.cnn.model import NetworkStitch
from darts.cnn import genotypes
cudnn.benchmark = True
cudnn.enabled=True
# from H_model_detone import H_estimator
import torch.nn as nn
import numpy as np

import cv2
import random
from align_attacks.evaluate_attacks_pgb import _fgsm_whitebox,_pgd_whitebox

from dataset import Image_stitch
import time

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
vis_batch = 50  # for checking loss during training
mode = 'supervise'  # optinal in ['supervise','unsupervise']
dataset_mode = 'mix'  # optinal in ['coco','roadscene','mix']
num_epochs = 100
learning_rate = 0.0002
height, width = 128, 128
batch_size = 8
data_root = '../data/training_roadandcoco_mix/'
# netR_path='snapshot/32_R.pkl'
# netH_path='snapshot/32_H.pkl'
netR_path = None
data = Image_stitch(ir1_path=os.path.join(data_root, 'ir_input1_coco'), \
                    ir2_path=os.path.join(data_root, 'ir_input2_coco'), \
                    vis1_path=os.path.join(data_root, 'vis_input1_coco'), \
                    vis2_path=os.path.join(data_root, 'vis_input2_coco'), \
                    gt_path=os.path.join(data_root, 'y_shift_coco'), \
                    mode=dataset_mode)
dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
l1loss = nn.L1Loss()
l2loss = nn.MSELoss()
genotype = eval("genotypes.%s" % 'UDIS')
netR = NetworkStitch(genotype=genotype, batch_size=batch_size)
#netR = H_estimator(batch_size=batch_size,device=device,is_training=1)
if netR_path is not None:
    netR.load_state_dict(torch.load(netR_path, map_location='cpu'))
if use_cuda:
    l1loss = l1loss.to(device)
    l2loss = l2loss.to(device)
    netR = netR.to(device)
optimizerR = torch.optim.Adam(netR.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
save_folder = 'snapshot_DARTS_coco'
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
for epoch in range(0, num_epochs + 1):
    for i, (ir_input1, ir_input2, vis_input1, vis_input2, size, gt, name) in enumerate(dataloader):

        # train_ir_inputs_aug = disjoint_augment_image_pair(ir_input1,ir_input2)
        train_vis_inputs_aug = disjoint_augment_image_pair(vis_input1, vis_input2)
        if use_cuda:
            # vis_input1=vis_input1.to(device)
            # vis_input2=vis_input2.to(device)
            train_vis_inputs_aug = train_vis_inputs_aug.to(device)
            size = size.to(device)
            gt = gt.to(device)

        # train_vis_inputs = torch.cat((vis_input1,vis_input2), 3)


        vis_off1, vis_off2, vis_off3, vis_warp1, vis_warp2 = netR(train_vis_inputs_aug, gt)
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
        ##### supervise training (optional)##################################################################################
        elif mode == 'supervise':
            vis_l2_gt1 = 0.02 * l2loss(gt, vis_off1)
            vis_l2_gt2 = 0.01 * l2loss(gt, vis_off1 + vis_off2)
            vis_l2_gt3 = 0.005 * l2loss(gt, vis_off1 + vis_off2 + vis_off3)
            loss_supervise = vis_l2_gt1 + vis_l2_gt2 + vis_l2_gt3
            loss_all = loss_supervise
            # loss_all = out_l2_gt
            l2_gt_1_batch += (vis_l2_gt1.item())
            l2_gt_2_batch += (vis_l2_gt2.item())
            l2_gt_3_batch += (vis_l2_gt3.item())
        loss_all_batch += loss_all.item()

        if i % vis_batch == 0 and i != 0:
            # print('shift: '+str(train_net1_f[0].reshape((1,8)).detach().cpu().numpy()))
            if mode == 'unsupervise':
                print(
                    '[%d/%d][%d/%d]\tlr_rate: %0.4f \tLoss_total: %.4f\t Loss_1: %.4f\t Loss_2: %.4f\t Loss_3: %.4f\t' %
                    (epoch, num_epochs, i, len(dataloader), optimizerR.state_dict()['param_groups'][0]['lr'],
                     loss_all_batch / vis_batch, l1_1_batch / vis_batch, l1_2_batch / vis_batch,
                     l1_3_batch / vis_batch))
                l1_1_batch = 0
                l1_2_batch = 0
                l1_3_batch = 0
            elif mode == 'supervise':
                print(
                    '[%d/%d][%d/%d]\tlr_rate: %0.4f \tLoss_total: %.4f\t Loss_1: %.4f\t Loss_2: %.4f\t Loss_3: %.4f\t' %
                    (epoch, num_epochs, i, len(dataloader), optimizerR.state_dict()['param_groups'][0]['lr'],
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
        optimizerR.zero_grad()
        # optimizerH.zero_grad()
    if epoch % 4 == 0:
        torch.save(netR.state_dict(), save_folder + '/' + str(epoch) + "_R.pkl")
        # torch.save(net_vis.state_dict(), save_folder+ '/' + str(epoch) + "_V.pkl")
        # torch.save(net_ir.state_dict(), save_folder+ '/' + str(epoch) + "_I.pkl")
        # torch.save(netH.state_dict(), save_folder+ '/' + str(epoch) + "_H.pkl")

