from __future__ import print_function
#%matplotlib inline
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import cv2
from darts.cnn import genotypes

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader
import torchvision
from dataset import Image_test
from visdom import Visdom
# from model import Reconstruction
import torch.nn.functional as F
from darts.cnn.model import Restructor
import PIL
import model
import cv2
import time

def load_checkpoint(model, optimizer, filename, logger, map_location):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=map_location)
        epoch = checkpoint.get('epoch', -1)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(map_location)
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return epoch


def environment_check():

    if torch.cuda.is_available():
        #os.system('gpustat')
        i = int(input("choose devise:"))

        if i != -1:
            torch.cuda.set_device(device=i)
            return
    print("cuda: False")

device=environment_check()
if device!='cpu':
    use_cuda=True
else:
    use_cuda=False
data_root_1 = '../output_COCO_256/std/'
#x = ['fgsm_train/','pgd_train/','bim_train/']
#x = ['one_layer/','two_layer/','four_layer/','five_layer/']
x = ['TIP/','JCVIR/']
y = ['clean', 'fgsm_attack/', 'pgd_attack/', 'bim_attack/']
for xi in x:
    for yi in y:
        #data_root='../output_UDIS/coco_train/std_TIP/bim_attack/'
        data_root = data_root_1 + xi + yi
        data = Image_test(0, os.path.join(data_root,'vis_warp1'),  \
                    os.path.join(data_root,'vis_warp2'), \
                    os.path.join(data_root,'vis_warp1'), \
                    os.path.join(data_root,'vis_warp2'))
        dataloader = torch.utils.data.DataLoader(data, batch_size=1,shuffle=False,num_workers=0)
        print(len(dataloader))
        if not os.path.exists(os.path.join(data_root,'vis_recon')):
            os.makedirs(os.path.join(data_root,'vis_recon'))
        if not os.path.exists(os.path.join(data_root,'ir_recon')):
            os.makedirs(os.path.join(data_root,'ir_recon'))

        genotype = eval("genotypes.%s" % 'unroll_true')

        #netG=Restructor(6,3,genotype)

        netG=model.UNet(6,3)
        #filename = 'coco/snapshot_Darts_630/8_G.pkl'
        filename = 'coco/snapshot_630/10_G.pkl'

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filename, map_location=device)
        """
        keys_to_remove = []
        for key in checkpoint.keys():
            if 'stem.1' in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del checkpoint[key]
        """
        netG.load_state_dict(checkpoint)

        # netG.load_state_dict(torch.load('snapshot/111_G.pkl', map_location='cpu'))
        if use_cuda:
            netG = netG.cuda()

        img_list = []
        iters = 0
        aa = 1
        i = 0
        print("Starting Training Loop...")
        # For each epoch
            # For each batch in the dataloader
        for i, (ir_warp1, ir_warp2,vis_warp1,vis_warp2,name) in enumerate(dataloader):
            if use_cuda:
                #ir_warp1 = ir_warp1.cuda()
                #ir_warp2 = ir_warp2.cuda()
                vis_warp1 = vis_warp1.cuda()
                vis_warp2 = vis_warp2.cuda()

            #ir_warp1=ir_warp1.float()
            #ir_warp2=ir_warp2.float()
            vis_warp1 = vis_warp1.float()
            vis_warp2 = vis_warp2.float()

            with torch.no_grad():
                #inputs_ir=torch.cat((ir_warp1,ir_warp2),1).float()
                #outputs_ir=netG(inputs_ir)
                inputs_vis=torch.cat((vis_warp1,vis_warp2),1).float()
                outputs_vis=netG(inputs_vis)

           #outputs_ir=(outputs_ir[0]).permute(1,2,0).detach().cpu().numpy()*255
            outputs_vis=(outputs_vis[0]).permute(1,2,0).detach().cpu().numpy()*255

            #name = str(name)

            #outputs_ir=cv2.cvtColor(outputs_ir, cv2.COLOR_BGR2RGB)
            outputs_vis=cv2.cvtColor(outputs_vis, cv2.COLOR_BGR2RGB)
            print(data_root,'vis_recon',name)
            for i in range(len(name)):
                #cv2.imwrite(os.path.join(data_root,'ir_recon',name[i]), outputs_ir)
                cv2.imwrite(os.path.join(data_root,'vis_recon',name[i]), outputs_vis)




