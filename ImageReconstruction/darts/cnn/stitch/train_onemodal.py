import os
from models import disjoint_augment_image_pair#,H_estimator
# from H_model_mini import H_estimator
from H_model import H_estimator,Feature_extractor,H_joint
# from H_model_detone import H_estimator
import torch.nn as nn
import numpy as np
import torch
import cv2
from dataset import Image_stitch
import time
from visdom import Visdom
# viz = Visdom(env='ZZX-stitch-pytorch')  # 启用可视化工具
# viz.line([0.], [0], win='total_loss', opts = dict(title='total_loss'))
# viz.line([0.], [0], win='l1_1_loss', opts = dict(title='l1_1_loss'))
# viz.line([0.], [0], win='l1_2_loss', opts = dict(title='l1_2_loss'))
# viz.line([0.], [0], win='l1_3_loss', opts = dict(title='l1_3_loss'))

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def environment_check():
    if torch.cuda.is_available():
        os.system('gpustat')
        i = int(input("choose devise:"))

        if i != -1:
            torch.cuda.set_device(device = i)
            return i
    print("cuda: False")
    return 'cpu'
device=environment_check()
if device!='cpu':
    use_cuda=True
else:
    use_cuda=False
vis_batch=50# for checking loss during training 
mode='supervise' #optinal in ['supervise','unsupervise']
num_epochs = 100
learning_rate = 0.0001
height, width = 128, 128
batch_size = 8
data_root='../../training_roadandcoco_mix/'
data=Image_stitch(ir1_path=os.path.join(data_root,'ir_input1'),\
                  ir2_path=os.path.join(data_root,'ir_input2'),\
                  vis1_path=os.path.join(data_root,'vis_input1'),\
                  vis2_path=os.path.join(data_root,'vis_input2'),\
                  gt_path=os.path.join(data_root,'y_shift'))
dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,shuffle=True,num_workers=12,pin_memory=True)
l1loss=nn.L1Loss()
l2loss=nn.MSELoss()

net_vis=Feature_extractor()
netR=H_estimator(batch_size=batch_size,device=device,is_training=1)
for m in net_vis.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_( m.weight )
if use_cuda:
    l1loss = l1loss.to(device)
    l2loss = l2loss.to(device)
    net_vis = net_vis.to(device)
    netR = netR.to(device)
optimizerV = torch.optim.Adam(net_vis.parameters(), lr = learning_rate, betas=(0.5, 0.999),weight_decay=0.0001)
optimizerR = torch.optim.Adam(netR.parameters(), lr = learning_rate, betas=(0.5, 0.999),weight_decay=0.0001)
save_folder = 'snapshot_onemodal'
# define dataset
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
loss_all_batch=0
if mode=='unsupervise':
    l1_1_batch =0
    l1_2_batch=0
    l1_3_batch =0
    l1_out_batch=0
elif mode=='supervise':
    l2_gt_1_batch=0
    l2_gt_2_batch=0
    l2_gt_3_batch=0
    l2_gt_out_batch=0
netR.train()
net_vis.train()
for epoch in range(0,num_epochs+1):
    # print(len(dataloader))
    # For each batch in the dataloader
    for i,(_,_,vis_input1,vis_input2,gt) in enumerate(dataloader):
        if use_cuda:
            vis_input1=vis_input1.to(device)
            vis_input2=vis_input2.to(device)
            gt=gt.to(device)
        train_vis_inputs_aug = disjoint_augment_image_pair(vis_input1,vis_input2)
        train_vis_inputs = torch.cat((vis_input1,vis_input2), 3)
        vis1_feature=net_vis(train_vis_inputs_aug[...,0:3].permute(0,3,1,2).float())
        vis2_feature=net_vis(train_vis_inputs_aug[...,3:6].permute(0,3,1,2).float())
        vis_off1, vis_off2, vis_off3, vis_warp2_H1, vis_warp2_H2, vis_warp2_H3, vis_one_warp_H1, vis_one_warp_H2, vis_one_warp_H3 , vis_warp2_gt, vis_one_warp_gt= netR(vis1_feature, vis2_feature,train_vis_inputs,gt)
        vis_shift=vis_off1+ vis_off2 +vis_off3
        ##### unsupervise training  (optional)##################################################################################
        if mode=='unsupervise':
            vis_l1_1 = 16*l1loss(vis_warp2_H1,  train_vis_inputs[...,0:3].permute(0,3,1,2)*vis_one_warp_H1)
            vis_l1_2 = 4*l1loss(vis_warp2_H2,  train_vis_inputs[...,0:3].permute(0,3,1,2)*vis_one_warp_H2)
            vis_l1_3 = 2*l1loss(vis_warp2_H3,  train_vis_inputs[...,0:3].permute(0,3,1,2)*vis_one_warp_H3)
            loss_unsupervise=ir_l1_1+ir_l1_2+ir_l1_3+ir_l1_4+vis_l1_1+vis_l1_2+vis_l1_3
            loss_all = loss_unsupervise
            l1_1_batch  +=  vis_l1_1.item()
            l1_2_batch  += vis_l1_2.item()
            l1_3_batch  +=  vis_l1_3.item()
        ##### supervise training (optional)##################################################################################
        elif mode=='supervise':
            vis_l2_gt1 = 0.02*l2loss(gt,vis_off1)
            vis_l2_gt2 = 0.01*l2loss(gt,vis_off1+vis_off2)
            vis_l2_gt3 = 0.005*l2loss(gt,vis_off1+vis_off2+vis_off3)
            loss_supervise = vis_l2_gt1+vis_l2_gt2+vis_l2_gt3#+out_l2_gt
            loss_all = loss_supervise
            l2_gt_1_batch += vis_l2_gt1.item()
            l2_gt_2_batch += vis_l2_gt2.item()
            l2_gt_3_batch += vis_l2_gt3.item()
        loss_all_batch += loss_all.item()
        
        if i % vis_batch == 0:
            # print('shift: '+str(train_net1_f[0].reshape((1,8)).detach().cpu().numpy()))
            if mode=='unsupervise':
                print('[%d/%d][%d/%d]\tlr_rate: %0.4f \tLoss_total: %.4f\t Loss_1: %.4f\t Loss_2: %.4f\t Loss_3: %.4f\t' % 
                    (epoch, num_epochs, i, len(dataloader),optimizerR.state_dict()['param_groups'][0]['lr'],loss_all_batch/vis_batch, l1_1_batch/vis_batch, l1_2_batch/vis_batch, l1_3_batch/vis_batch))
                l1_1_batch =0
                l1_2_batch=0
                l1_3_batch =0
            elif mode=='supervise':
                print('[%d/%d][%d/%d]\tlr_rate: %0.4f \tLoss_total: %.4f\t Loss_1: %.4f\t Loss_2: %.4f\t Loss_3: %.4f\t' % 
                    (epoch, num_epochs, i, len(dataloader),optimizerR.state_dict()['param_groups'][0]['lr'],loss_all_batch/vis_batch, l2_gt_1_batch/vis_batch, l2_gt_2_batch/vis_batch, l2_gt_3_batch/vis_batch))
                l2_gt_1_batch=0
                l2_gt_2_batch=0
                l2_gt_3_batch=0
            # viz.line([loss_all.item()], [epoch + (i + 1) / len(dataloader)], win='total_loss', update='append')
            # viz.line([l1_1.item()], [epoch + (i + 1) / len(dataloader)], win='l1_1_loss', update='append')
            # viz.line([l1_2.item()], [epoch + (i + 1) / len(dataloader)], win='l1_2_loss', update='append')
            # viz.line([l1_3.item()], [epoch + (i + 1) / len(dataloader)], win='l1_3_loss', update='append')
            # viz.line([loss_all_batch/25], [epoch + (i + 1) / len(dataloader)], win='total_loss', update='append')
            # viz.line([l1_1_batch/25], [epoch + (i + 1) / len(dataloader)], win='l1_1_loss', update='append')
            # viz.line([l1_2_batch/25], [epoch + (i + 1) / len(dataloader)], win='l1_2_loss', update='append')
            # viz.line([l1_3_batch./25], [epoch + (i + 1) / len(dataloader)], win='l1_3_loss', update='append')
            loss_all_batch=0
            

        if i % vis_batch == 0:
            vis_warp1 = torch.cat(((train_vis_inputs[0][...,0:3].permute(2,0,1) * vis_one_warp_H1[0]),
                                  (train_vis_inputs[0][...,0:3].permute(2,0,1) * vis_one_warp_H2[0]),
                                  (train_vis_inputs[0][...,0:3].permute(2,0,1) * vis_one_warp_H3[0])),1)
            vis_warp2 = torch.cat((vis_warp2_H1[0], vis_warp2_H2[0], vis_warp2_H3[0]),1)
            vis_warp=torch.cat((vis_warp1,vis_warp2),2)
            vis_input1_ = vis_input1[0].permute(2,0,1)
            vis_input2_ = vis_input2[0].permute(2,0,1)
            vis_one_warp_gt_ = vis_one_warp_gt[0] * train_vis_inputs[0][...,0:3].permute(2,0,1)
            vis_input_gt_ = torch.cat((vis_one_warp_gt_,vis_warp2_gt[0]),2)
            vis_input_ = torch.cat((vis_input1_,vis_input2_),2)
            viss=torch.cat((vis_input_,vis_warp,vis_input_gt_),1).permute(1,2,0).detach().cpu().numpy()*255
            cv2.imwrite('visual_onemodal.png',viss)
        loss_all.backward()
        optimizerR.step()
        # optimizerI.step()
        optimizerV.step()
        # optimizerH.step()
        optimizerR.zero_grad()
        # optimizerI.zero_grad()
        optimizerV.zero_grad()
        # optimizerH.zero_grad()
    if epoch % 2 == 0:
        torch.save(netR.state_dict(), save_folder+ '/' + str(epoch) + "_R.pkl")
        torch.save(net_vis.state_dict(), save_folder+ '/' + str(epoch) + "_V.pkl")
