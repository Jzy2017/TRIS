import os
from models import disjoint_augment_image_pair#,H_estimator
# from H_model_mini import H_estimator
from H_model import H_estimator,H_joint_out, H_joint
# from H_model_detone import H_estimator
import torch.nn as nn
import numpy as np
import torch
import cv2
from dataset import Image_stitch
from align_attacks.evaluate_attacks_pgb import _fgsm_whitebox
import time
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
dataset_mode='mix'
def environment_check():

    if torch.cuda.is_available():
        #os.system('gpustat')
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
netR_path='snapshot/20_R.pkl'
netH_path='snapshot/20_H.pkl'
batch_size = 1
data_root = '../data/Roadscenetest'
out_folder = os.path.join('../output/',data_root.split('/')[-1])
data=Image_stitch(ir1_path=os.path.join(data_root,'ir_input1'),\
                  ir2_path=os.path.join(data_root,'ir_input2'),\
                  vis1_path=os.path.join(data_root,'vis_input1'),\
                  vis2_path=os.path.join(data_root,'vis_input2'),\
                  gt_path=os.path.join(data_root,'y_shift'),\
                  mode=dataset_mode)
dataloader = torch.utils.data.DataLoader(data, batch_size= batch_size,shuffle=False,num_workers=0,pin_memory=True)

netR=H_estimator(batch_size=batch_size,device=device,is_training=False)
netH=H_joint_out(batch_size=batch_size,device=device,is_training=False)
netH_attack = H_joint(batch_size=batch_size,device=device,is_training=False)
if netR_path is not None:
    netR.load_state_dict(torch.load(netR_path,map_location='cpu'))
if netH_path is not None:
    netH.load_state_dict(torch.load(netH_path,map_location='cpu'))
    netH_attack.load_state_dict(torch.load(netH_path,map_location='cpu'))
if use_cuda:
    netR = netR.to(device)
    netH = netH.to(device)
    netH_attack = netH_attack.to(device)

# define dataset
if not os.path.exists(os.path.join(out_folder,'ir_warp1')):
    os.makedirs(os.path.join(out_folder,'ir_warp1'))
if not os.path.exists(os.path.join(out_folder,'ir_warp2')):
    os.makedirs(os.path.join(out_folder,'ir_warp2'))
if not os.path.exists(os.path.join(out_folder,'vis_warp1')):
    os.makedirs(os.path.join(out_folder,'vis_warp1'))
if not os.path.exists(os.path.join(out_folder,'vis_warp2')):
    os.makedirs(os.path.join(out_folder,'vis_warp2'))
if not os.path.exists(os.path.join(out_folder, 'ir_warp_combine')):
    os.makedirs(os.path.join(out_folder, 'ir_warp_combine'))
if not os.path.exists(os.path.join(out_folder, 'vis_warp_combine')):
    os.makedirs(os.path.join(out_folder, 'vis_warp_combine'))
"""

output attacked image

"""
if not os.path.exists(os.path.join(out_folder,'ir_attack1')):
    os.makedirs(os.path.join(out_folder,'ir_attack1'))
if not os.path.exists(os.path.join(out_folder,'ir_attack2')):
    os.makedirs(os.path.join(out_folder,'ir_attack2'))
if not os.path.exists(os.path.join(out_folder,'vis_attack1')):
    os.makedirs(os.path.join(out_folder,'vis_attack1'))
if not os.path.exists(os.path.join(out_folder, 'vis_attack2')):
    os.makedirs(os.path.join(out_folder, 'vis_attack2'))

loss_all_batch = 0
l1_1_batch = 0
l1_2_batch = 0
l1_3_batch = 0
l2_gt_batch = 0
netR.eval()
netH.eval()
netH_attack.eval()

for i,(ir_input1,ir_input2,vis_input1,vis_input2,size, gt, name) in enumerate(dataloader):
    print(name[0])
    #train_ir_inputs_aug = disjoint_augment_image_pair(ir_input1, ir_input2)
    #train_vis_inputs_aug = disjoint_augment_image_pair(vis_input1, vis_input2)
    if use_cuda:
        ir_input1=ir_input1.to(device)
        ir_input2=ir_input2.to(device)
        vis_input1=vis_input1.to(device)
        vis_input2=vis_input2.to(device)
        size=size.to(device)
        gt = gt.to(device)
    train_ir_inputs=torch.cat((ir_input1,ir_input2), 3)
    train_vis_inputs=torch.cat((vis_input1,vis_input2), 3)
    #print(train_vis_inputs.shape)
    train_ir_inputs, train_vis_inputs = _fgsm_whitebox(netR, netH_attack, train_ir_inputs, train_ir_inputs,
                                                       train_vis_inputs, train_vis_inputs, size, gt)

    # save attacked image
    #print(train_ir_inputs.shape)
    ir_inputs1_attack = (train_ir_inputs.squeeze(0)[...,:3].detach().cpu().numpy()*255).astype(np.uint8)
    ir_inputs2_attack = (train_ir_inputs.squeeze(0)[...,3:6].detach().cpu().numpy()*255).astype(np.uint8)
    vis_inputs1_attack = (train_vis_inputs.squeeze(0)[...,:3].detach().cpu().numpy()*255).astype(np.uint8)
    vis_inputs2_attack = (train_ir_inputs.squeeze(0)[...,3:6].detach().cpu().numpy()*255).astype(np.uint8)

    #cv2.imwrite(os.path.join(out_folder,'ir_attack1',name[0]),cv2.cvtColor(ir_inputs1_attack,cv2.COLOR_RGB2BGR))
    #cv2.imwrite(os.path.join(out_folder,'ir_attack2',name[0]),cv2.cvtColor(ir_inputs2_attack,cv2.COLOR_RGB2BGR))
    #cv2.imwrite(os.path.join(out_folder,'vis_attack1',name[0]),cv2.cvtColor(vis_inputs1_attack,cv2.COLOR_RGB2BGR))
    #cv2.imwrite(os.path.join(out_folder,'vis_attack2',name[0]),cv2.cvtColor(vis_inputs2_attack,cv2.COLOR_RGB2BGR))

    #print(train_vis_inputs.shape)
    with torch.no_grad():
        ir_off, vis_off = netR(None,train_ir_inputs,None,train_vis_inputs,gt=None,is_test=True)
        warps_ir, warps_vis = netH(ir_off,vis_off, size,train_ir_inputs,train_vis_inputs)
       

    ir_warp1_H=warps_ir[0][0:3].permute(1,2,0).detach().cpu().numpy()*255
    ir_warp2_H=warps_ir[0][3:6].permute(1,2,0).detach().cpu().numpy()*255
    vis_warp1_H=warps_vis[0][0:3].permute(1,2,0).detach().cpu().numpy()*255
    vis_warp2_H=warps_vis[0][3:6].permute(1,2,0).detach().cpu().numpy()*255

    # 从图1取0通道作为新图的0通道
    new_img_0 = ir_warp1_H[:, :, 0]
    # 从图2取2通道作为新图的2通道
    new_img_2 = ir_warp2_H[:, :, 2]
    # 从图1的1通道，图2的1通道相加除以2作为新图的1通道
    new_img_1 = (ir_warp1_H[:, :, 1] + ir_warp2_H[:, :, 1]) / 2
    # 合并三个通道以创建新图像
    new_img = np.stack([new_img_0, new_img_1, new_img_2], axis=2)
    cv2.imwrite(os.path.join(out_folder, 'ir_warp_combine', name[0]), cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))

    # 从图1取0通道作为新图的0通道
    new_img_0 = vis_warp1_H[:, :, 0]
    # 从图2取2通道作为新图的2通道
    new_img_2 = vis_warp2_H[:, :, 2]
    # 从图1的1通道，图2的1通道相加除以2作为新图的1通道
    new_img_1 = (vis_warp1_H[:, :, 1] + vis_warp2_H[:, :, 1]) / 2
    # 合并三个通道以创建新图像
    new_img = np.stack([new_img_0, new_img_1, new_img_2], axis=2)
    cv2.imwrite(os.path.join(out_folder, 'vis_warp_combine', name[0]), cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))

    cv2.imwrite(os.path.join(out_folder,'ir_warp1',name[0]),cv2.cvtColor(ir_warp1_H,cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_folder,'ir_warp2',name[0]),cv2.cvtColor(ir_warp2_H,cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_folder,'vis_warp1',name[0]),cv2.cvtColor(vis_warp1_H,cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_folder,'vis_warp2',name[0]),cv2.cvtColor(vis_warp2_H,cv2.COLOR_RGB2BGR))

