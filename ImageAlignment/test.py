import os
from models import disjoint_augment_image_pair#,H_estimator
# from H_model_mini import H_estimator
from H_model import H_estimator, H_joint_out
from darts.cnn.model import NetworkStitch
from darts.cnn import genotypes

# from H_model_detone import H_estimator
import torch.nn as nn
import numpy as np
import torch
import cv2
from dataset import Image_stitch, Image_UDIS
from align_attacks.evaluate_attacks_pgb import _fgsm_whitebox, _pgd_whitebox, _bim_whitebox
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
netR_path='snapshot_std_coco/20_R.pkl'
batch_size = 1
data_root='../data/challenge_gen'
#out_folder = '../output_UDIS/multi_scale/one_scale/'
"""
data=Image_UDIS(vis1_path=os.path.join(data_root,'input1'),\
                  vis2_path=os.path.join(data_root,'input2'),
                is_train=False)"""
data=Image_UDIS(vis1_path=os.path.join(data_root,'input1'),\
                  vis2_path=os.path.join(data_root,'input2'),
                is_train=False)
dataloader = torch.utils.data.DataLoader(data, batch_size= batch_size,shuffle=False,num_workers=0,pin_memory=True)

genotype = eval("genotypes.%s" % 'UDIS')
#print(genotype)
netR = NetworkStitch(genotype=genotype, batch_size=batch_size)

#netR=H_estimator(batch_size=batch_size,device=device,is_training=False)
netH = H_joint_out()
#pth_bin = ['snapshot_std_UDIS_2/40_R.pkl','snapshot_std_coco/40_R.pkl', 'snapshot_std_UDIS/28_R.pkl', 'snapshot_fgsm_train/snapshot_fgsm_train/40_R.pkl', 'snapshot_pgd_train/snapshot_pgd_train/40_R.pkl', 'snapshot_bim_train/40_R.pkl']
pth_bin = ['snapshot_DARTS_for_our_attack/40_R.pkl']
#['snapshot_fgsm_train/snapshot_fgsm_train/40_R.pkl','snapshot_pgd_train/snapshot_pgd_train/40_R.pkl','snapshot_bim_train/40_R.pkl']
for netR_path in pth_bin:
    if netR_path is not None:
        netR.load_state_dict(torch.load(netR_path,map_location='cpu'))
    if use_cuda:
        netR = netR.to(device)
    #'clean','fgsm_attack',
    for attack in ['bim_attack']:
        #attack = 'fgsm_attack'
        if netR_path == 'snapshot_fgsm_train/snapshot_fgsm_train/40_R.pkl':
            out_folder = '../output_challenge/adversarial_train/fgsm_train/'

        if netR_path == 'snapshot_pgd_train/snapshot_pgd_train/40_R.pkl':
            out_folder = '../output_challenge/adversarial_train/pgd_train/'

        if netR_path == 'snapshot_bim_train/40_R.pkl':
            out_folder = '../output_challenge/adversarial_train/bim_train/'

        if netR_path == 'snapshot_std_UDIS/28_R.pkl':
            out_folder = '../output_challenge/std/JCVIR/'

        if netR_path == 'snapshot_DARTS_UDIS/44_R.pkl':
            out_folder = '../output_challenge/adversarial_train/Ours/'

        if netR_path == 'snapshot_DARTS_for_our_attack/40_R.pkl':
            out_folder = '../output_challenge/adversarial_train/Ours_our_attack/'

        if netR_path == 'snapshot_std_coco/40_R.pkl':
            out_folder = '../output_challenge/multi_scale/five_scale/'

        if netR_path == 'snapshot_std_UDIS_2/40_R.pkl':
            out_folder = '../output_challenge/std/TIP/'

        if netR_path == 'snapshot_DARTS_challenge/40_R.pkl':
            out_folder = '../output_challenge/adversarial_train/Ours3/'

        if netR_path == 'snapshot_only_for_clean/40_R.pkl':
            out_folder = '../output_challenge/std/Ours/'

        out_folder = out_folder + attack + '/'
        # define dataset
        if not os.path.exists(os.path.join(out_folder,'vis_warp1')):
            os.makedirs(os.path.join(out_folder,'vis_warp1'))
        if not os.path.exists(os.path.join(out_folder,'vis_warp2')):
            os.makedirs(os.path.join(out_folder,'vis_warp2'))
        if not os.path.exists(os.path.join(out_folder, 'vis_warp_combine')):
            os.makedirs(os.path.join(out_folder, 'vis_warp_combine'))
        if not os.path.exists(os.path.join(out_folder, 'mask1')):
            os.makedirs(os.path.join(out_folder, 'mask1'))
        if not os.path.exists(os.path.join(out_folder, 'mask2')):
            os.makedirs(os.path.join(out_folder, 'mask2'))
        """
        
        output attacked image
        
        """
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

        for i,(vis_input1,vis_input2,size,name) in enumerate(dataloader):
            print(name[0])
            #train_ir_inputs_aug = disjoint_augment_image_pair(ir_input1, ir_input2)
            #train_vis_inputs_aug = disjoint_augment_image_pair(vis_input1, vis_input2)
            if use_cuda:
                vis_input1=vis_input1.to(device)
                vis_input2=vis_input2.to(device)
                size=size.to(device)
            #print(vis_input1.shape, vis_input2.shape)
            train_vis_inputs=torch.cat((vis_input1,vis_input2), 3)
            #print(train_vis_inputs.shape)

            if attack == 'fgsm_attack':
                train_vis_inputs = _fgsm_whitebox(netR, train_vis_inputs)
            elif attack == 'pgd_attack':
                train_vis_inputs = _pgd_whitebox(netR, train_vis_inputs)
            elif attack == 'our_attack':
                train_vis_inputs = _pgd_whitebox(netR, train_vis_inputs)
            elif attack == 'bim_attack':
                train_vis_inputs = _bim_whitebox(netR, train_vis_inputs)
            else:
                train_vis_inputs = train_vis_inputs

            #print(train_ir_inputs.shape)
            vis_inputs1_attack = (train_vis_inputs.squeeze(0)[...,:3].detach().cpu().numpy()*255).astype(np.uint8)
            vis_inputs2_attack = (train_vis_inputs.squeeze(0)[...,3:6].detach().cpu().numpy()*255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_folder,'vis_attack1',name[0]),cv2.cvtColor(vis_inputs1_attack,cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_folder,'vis_attack2',name[0]),cv2.cvtColor(vis_inputs2_attack,cv2.COLOR_RGB2BGR))

            #print(train_vis_inputs.shape)
            with torch.no_grad():
                vis_off = netR(train_vis_inputs,gt=None,is_test=True)
                #print(vis_off.shape)
                warps_vis = netH(vis_off, size, train_vis_inputs)
            vis_warp1_H=warps_vis[0][0:3].permute(1,2,0).detach().cpu().numpy()*255
            vis_warp2_H=warps_vis[0][3:6].permute(1,2,0).detach().cpu().numpy()*255
            mask1 = warps_vis[0][6:9].permute(1,2,0).detach().cpu().numpy()*255
            mask2 = warps_vis[0][9:12].permute(1, 2, 0).detach().cpu().numpy() * 255
            # 从图1取0通道作为新图的0通道
            new_img_0 = vis_warp1_H[:, :, 0]
            # 从图2取2通道作为新图的2通道
            new_img_2 = vis_warp2_H[:, :, 2]
            # 从图1的1通道，图2的1通道相加除以2作为新图的1通道
            new_img_1 = (vis_warp1_H[:, :, 1] + vis_warp2_H[:, :, 1]) / 2
            # 合并三个通道以创建新图像
            new_img = np.stack([new_img_0, new_img_1, new_img_2], axis=2)
            cv2.imwrite(os.path.join(out_folder, 'vis_warp_combine', name[0]), cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))

            cv2.imwrite(os.path.join(out_folder,'vis_warp1',name[0]),cv2.cvtColor(vis_warp1_H,cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_folder,'vis_warp2',name[0]),cv2.cvtColor(vis_warp2_H,cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_folder, 'mask1', name[0]), cv2.cvtColor(mask1, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_folder, 'mask2', name[0]), cv2.cvtColor(mask2, cv2.COLOR_RGB2BGR))

