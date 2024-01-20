from __future__ import print_function
#%matplotlib inline
import os
import torch
from tqdm import tqdm
import torch.nn as nn
# import torch.nn.parallel
# import torch.utils.data
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader
import torchvision
from dataset import Image
from visdom import Visdom
# from model import Reconstruction
import torch.nn.functional as F
#from generator import Generator
#sys.path.append(r"src/model")
import loss
import PIL
import model
# import utils
import time
train_augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224),interpolation=PIL.Image.NEAREST)])
train_augmentation_image = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224),interpolation=PIL.Image.BILINEAR)])
batch_size=4
num_epochs=200
learning_rate=0.0001
#os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = 'cpu'
# viz = Visdom(env='ReConstruction')  #启用可视化工具
# viz = Visdom(env='ZZX_IFCNN_one')  #启用可视化工具
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
nz = 200
data_root='../output/onemodalalign/fgsm_attack'
data = Image(0, os.path.join(data_root,'vis_warp1'),  \
            os.path.join(data_root,'vis_warp2'), \
            os.path.join(data_root,'mask1'), \
            os.path.join(data_root,'mask2'))
dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,shuffle=True,num_workers=0)
print(len(dataloader))
# Decide which device we want to run on
# device = torch.device("cuda:1" if (torch.cuda.is_available() and use_cuda) else "cpu")
# utils.environment_check()  #检查环境用的
ssim_val = loss.SSIM()
mse_loss = nn.MSELoss()
per_loss=loss.PerceptualLoss()
l1_loss=nn.L1Loss()
if use_cuda:
    ssim_val = ssim_val.cuda()
    mse_loss = mse_loss.cuda()
    l1_loss=l1_loss.cuda()
    per_loss=per_loss.cuda()


#viz.line([0.], [0], win='seam_loss', opts=dict(title='seam_loss'))
#viz.line([0.], [0], win='content_loss', opts=dict(title='content_loss'))

#viz.line([0.], [0], win='total_loss', opts=dict(title='total_loss'))



netG=model.UNet(6,3)
netAff = model.LocNet()
# netG.load_state_dict(torch.load('snapshot_noada/48_G.pkl', map_location='cpu'))
if use_cuda:
    netG = netG.cuda()
    # netAff=netAff.cuda()

optimizer = torch.optim.Adam(netG.parameters(), lr=learning_rate/2, betas=(0.5, 0.999),weight_decay=0.0001)

# Training Loop

# Lists to keep track of progress
img_list = []
# G_losses = []
iters = 0
aa = 1
i = 0
print("Starting Training Loop...")
# For each epoch
for epoch in range(1,250):
    # For each batch in the dataloader
    # for i,(warp1,warp2,mask1,mask2,label2) in enumerate(dataloader):
    for i,(warp1,warp2,mask1,mask2) in enumerate(dataloader):
        optimizer.state_dict()['param_groups'][0]['lr']=optimizer.state_dict()['param_groups'][0]['lr']/2.
        netG.zero_grad()
        if use_cuda:
            warp1=warp1.cuda()
            warp2=warp2.cuda()
            mask1=mask1.cuda()
            mask2=mask2.cuda()
            #label2=label2.cuda()
        
        warp1 = warp1.float()
        warp2 = warp2.float()
        mask1 = mask1.float()
        mask2 = mask2.float()
        # label2 = label2.float()

        inputs=torch.cat((warp1,warp2),1)
        outputs=netG(inputs)

        mask1_224=train_augmentation(mask1)

        seam_mask1 = mask1*model.seammask_extraction(mask2,use_cuda)
        seam_mask2 = mask2*model.seammask_extraction(mask1,use_cuda)
        seam_loss1 = l1_loss(outputs*seam_mask1, warp1*seam_mask1)
        seam_loss2 = l1_loss(outputs*seam_mask2, warp2*seam_mask2)
        train_stitched1 = train_augmentation_image(outputs*mask1)  
        train_stitched2 = train_augmentation_image(outputs*mask2)  
        train_warp1 = train_augmentation_image(warp1*mask1)
        train_warp2 = train_augmentation_image(warp2*mask2)     
        content_loss1 = per_loss(train_stitched1,train_warp1 )
        content_loss2  = per_loss(train_stitched2,train_warp2 )

        # total low-resolution reconstruction loss
        seam_loss=(seam_loss1 + seam_loss2)*2
        content_loss= (content_loss1+content_loss2)
        # # Loss
        loss_all = 1*seam_loss + 0.01*content_loss#+1*sr_loss+Aff_loss
        loss_all.backward()
        optimizer.step()
        # Output training stats
        if i % 100 == 0:
            print('[%d/%d][%d/%d]\tlr_rate: %0.4f \tLoss_total: %.4f\tLoss_seam: %.4f\tLoss_content: %0.4f\t' %
                  (epoch, num_epochs, i, len(dataloader),optimizer.state_dict()['param_groups'][0]['lr'],loss_all.mean().item(), seam_loss.mean().item(),0.001*content_loss.mean().item()))
        """
        if i % 50 == 0:
            viz.line([loss_all.mean().item()], [epoch + (i + 1) / len(dataloader)], win='total_loss', update='append')
            viz.line([seam_loss.mean().item()], [epoch + (i + 1) / len(dataloader)], win='seam_loss', update='append')
            viz.line([0.001*content_loss.mean().item()], [epoch + (i + 1) / len(dataloader)], win='content_loss', update='append')
            viz.line([content_loss.mean().item()], [epoch + (i + 1) / len(dataloader)], win='sr_loss', update='append')
        """
        # if i % 50 == 0:
        #     if batch_size>4:
        #         viz.images(out[0:4], nrow=4, win='heatmaps', opts={'title': 'headmaps'})
        #         viz.images(gt[0:4], nrow=4, win='gtmaps', opts={'title': 'gtmaps'})
        #     else:
        #         viz.images(out, nrow=1, win='heatmaps', opts={'title': 'headmaps'})
        #         viz.images(gt, nrow=1, win='gtmaps', opts={'title': 'gtmaps'})
        #Check how the generator is doing by saving G's output on fixed_noise
    if epoch % 2 == 0:
        torch.save(netG.state_dict(), 'snapshot_fgsm_attack/' + str(epoch) + "_G.pkl")
        # torch.save(netAff.state_dict(), 'snapshot/' + str(epoch)+ "_aff.pkl")

