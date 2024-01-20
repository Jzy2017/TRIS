import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
#from lib.losses.loss_function import GupnetLoss
import cv2

l1loss=nn.L1Loss()
l2loss=nn.MSELoss()

def calculate_overlap(warp1, warp2):
    # 将warp图像转化为二值图像，我们假设大于0的像素为有效像素
    warp1_binary = (warp1 > 0).float()
    warp2_binary = (warp2 > 0).float()

    # 计算两个二值图像的重叠部分
    overlap = warp1_binary * warp2_binary

    # 计算重叠部分的面积（即像素数）
    overlap_area = torch.sum(overlap, dim=[1, 2, 3])

    return overlap_area

def _pgd_whitebox(net,
                  X,
                  epsilon=8 / 255.0,
                  num_steps=3,
                  step_size=3 / 255.0):
    X_pgd = Variable(X.data, requires_grad=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            vis_off1, vis_off2, vis_off3, vis_warp1, vis_warp2 = net(X_pgd)
            vis_l1_1 = 16 * l1loss(vis_warp1[:, 0:3, ...], vis_warp2[:, 0:3, ...])
            vis_l1_2 = 4 * l1loss(vis_warp1[:, 3:6, ...], vis_warp2[:, 3:6, ...])
            vis_l1_3 = 2 * l1loss(vis_warp1[:, 6:9, ...], vis_warp2[:, 6:9, ...])
            loss = vis_l1_1 + vis_l1_2 + vis_l1_3

        grad_x = torch.autograd.grad(loss, X_pgd)
        eta = step_size * grad_x[0].data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    with torch.enable_grad():
        vis_off1, vis_off2, vis_off3, vis_warp1, vis_warp2 = net(X)
        vis_off1_pgd, vis_off2_pgd, vis_off3_pgd, vis_warp1_pgd, vis_warp2_pgd = net(X_pgd)
        overlap_area = calculate_overlap(vis_warp1, vis_warp2) - calculate_overlap(vis_warp1_pgd, vis_warp2_pgd)
        loss_overlap = 0.01 * torch.sum(overlap_area ** 2)
        loss_off = 0.01 * l2loss(vis_off1+vis_off2+vis_off3, vis_off1_pgd+vis_off2_pgd+vis_off3_pgd)
        loss = loss_overlap + loss_off

    grad_x = torch.autograd.grad(loss, X_pgd)
    eta = step_size * grad_x[0].data.sign()
    X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
    eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
    X_pgd = Variable(X.data + eta, requires_grad=True)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    return X_pgd


def _bim_whitebox(net,
                  X,
                  epsilon=8 / 255.0,
                  num_steps=2,
                  step_size=3 / 255.0):
    X_pgd = Variable(X.data, requires_grad=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    #X_pgd = Variable(X_pgd.data, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            vis_off1, vis_off2, vis_off3, vis_warp1, vis_warp2 = net(X_pgd)
            vis_l1_1 = 16 * l1loss(vis_warp1[:, 0:3, ...], vis_warp2[:, 0:3, ...])
            vis_l1_2 = 4 * l1loss(vis_warp1[:, 3:6, ...], vis_warp2[:, 3:6, ...])
            vis_l1_3 = 2 * l1loss(vis_warp1[:, 6:9, ...], vis_warp2[:, 6:9, ...])
            loss = vis_l1_1 + vis_l1_2 + vis_l1_3

        grad_x = torch.autograd.grad(loss, X_pgd)
        eta = step_size * grad_x[0].data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    return X_pgd


def _fgsm_whitebox(net,
                   train_vis_inputs,
                   epsilon=(8 / 255.0)):
    #model.eval()
    train_vis_inputs_fgsm = Variable(train_vis_inputs.data, requires_grad=True)
    train_vis_inputs_fgsm.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opt = optim.SGD([train_vis_inputs_fgsm], lr=1e-3)
    opt.zero_grad()

    with torch.enable_grad():
        vis_off1, vis_off2, vis_off3, vis_warp1, vis_warp2= net(train_vis_inputs_fgsm)
        vis_l1_1 = 16 * l1loss(vis_warp1[:, 0:3, ...], vis_warp2[:, 0:3, ...])
        vis_l1_2 = 4 * l1loss(vis_warp1[:, 3:6, ...], vis_warp2[:, 3:6, ...])
        vis_l1_3 = 2 * l1loss(vis_warp1[:, 6:9, ...], vis_warp2[:, 6:9, ...])
        loss = vis_l1_1 + vis_l1_2 + vis_l1_3


    grad_x = torch.autograd.grad(loss, train_vis_inputs_fgsm)

    #loss_supervise.backward()
    train_vis_inputs_fgsm = Variable(torch.clamp(train_vis_inputs_fgsm.data + epsilon * grad_x[0].data.sign(), 0.0, 1.0), requires_grad=False)

    #model.train
    return train_vis_inputs_fgsm

