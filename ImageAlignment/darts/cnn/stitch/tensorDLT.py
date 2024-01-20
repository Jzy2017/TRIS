# import tensorflow.compat.v1 as tf
import numpy as np
import torch
import time
#######################################################
# Auxiliary matrices used to solve DLT
class tensor_DLT(torch.nn.Module):
    def __init__(self,batch_size):
        super().__init__()
        # M1_tensor = torch.unsqueeze(self.M1, 0)
        self.M1_tile  = torch.unsqueeze(torch.tensor(np.array([
                [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [ 1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [ 0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
                [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [ 0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
                [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [ 0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)).float(), 0).repeat([batch_size ,1 ,1])


        self.M2_tile  = torch.unsqueeze(torch.tensor(np.array([
                [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [ 0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [ 0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
                [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [ 0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
                [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ]], dtype=np.float64)).float(), 0).repeat([batch_size ,1 ,1])



        self.M3_tile  = torch.unsqueeze(torch.tensor(np.array([
                [0],
                [1],
                [0],
                [1],
                [0],
                [1],
                [0],
                [1]], dtype=np.float64)).float(), 0).repeat([batch_size ,1 ,1])



        self.M4_tile  = torch.unsqueeze(torch.tensor(np.array([
                [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)).float(), 0).repeat([batch_size ,1 ,1])


        self.M5_tile  = torch.unsqueeze(torch.tensor(np.array([
                [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float64)).float(), 0).repeat([batch_size ,1 ,1])



        self.M6_tile  = torch.unsqueeze(torch.tensor(np.array([
                [-1 ],
                [ 0 ],
                [-1 ],
                [ 0 ],
                [-1 ],
                [ 0 ],
                [-1 ],
                [ 0 ]], dtype=np.float64)).float(), 0).repeat([batch_size ,1 ,1])


        self.M71_tile  = torch.unsqueeze(torch.tensor(np.array([
                [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)).float(), 0).repeat([batch_size ,1 ,1])


        self.M72_tile  = torch.unsqueeze(torch.tensor(np.array([
                [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ]], dtype=np.float64)).float(), 0).repeat([batch_size ,1 ,1])



        self.M8_tile  = torch.unsqueeze(torch.tensor(np.array([
                [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ]], dtype=np.float64)).float(), 0).repeat([batch_size ,1 ,1])


        self.Mb_tile  = torch.unsqueeze(torch.tensor(np.array([
                [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 , -1  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
                [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float64)).float(),0).repeat([batch_size ,1 ,1])
        self.pts_1_tile = torch.tensor([0., 0., 128., 0., 0., 128., 128., 128.]).unsqueeze(1).unsqueeze(0).float().repeat([batch_size, 1, 1])
        self.h_ones = torch.ones([batch_size, 1, 1])
        ########################################################
    def to(self, device='cpu'):
        self.M1_tile = self.M1_tile.to(device)
        self.M2_tile = self.M2_tile.to(device)
        self.M3_tile = self.M3_tile.to(device)
        self.M4_tile = self.M4_tile.to(device)
        self.M5_tile = self.M5_tile.to(device)
        self.M6_tile = self.M6_tile.to(device)
        self.M71_tile = self.M71_tile.to(device)
        self.M72_tile = self.M72_tile.to(device)
        self.M8_tile = self.M8_tile.to(device)
        self.Mb_tile = self.Mb_tile.to(device)
        self.h_ones = self.h_ones.to(device)
        self.pts_1_tile = self.pts_1_tile.to(device)
        return self
    def forward(self, pre_4pt_shift,patch_size=128.):

        batch_size = pre_4pt_shift.shape[0]
        # print(self.pts_1_tile.device)
        # print(pre_4pt_shift.device)
        # time.sleep(1000)
        pred_pts_2_tile = pre_4pt_shift + self.pts_1_tile/(128./patch_size)
        # change the order to get the inverse matrix
        orig_pt4 = pred_pts_2_tile
        pred_pt4 = self.pts_1_tile/(128./patch_size)

        

        # Form the equations Ax = b to compute H
        # Form A matrix
        A1 = torch.matmul(self.M1_tile, orig_pt4) # Column 1
        A2 = torch.matmul(self.M2_tile, orig_pt4) # Column 2
        A3 = self.M3_tile                   # Column 3
        A4 = torch.matmul(self.M4_tile, orig_pt4) # Column 4
        A5 = torch.matmul(self.M5_tile, orig_pt4) # Column 5
        A6 = self.M6_tile                   # Column 6
        A7 = torch.matmul(self.M71_tile, pred_pt4) *  torch.matmul(self.M72_tile, orig_pt4  )# Column 7
        A8 = torch.matmul(self.M71_tile, pred_pt4) *  torch.matmul(self.M8_tile, orig_pt4  )# Column 8

        # tmp = tf.reshape(A1, [-1, 8])  #batch_size * 8
        # A_mat: batch_size * 8 * 8    
        a=torch.stack([torch.reshape(A1 ,[-1 ,8]) ,torch.reshape(A2 ,[-1 ,8]), \
                                    torch.reshape(A3 ,[-1 ,8]) ,torch.reshape(A4 ,[-1 ,8]), \
                                    torch.reshape(A5 ,[-1 ,8]) ,torch.reshape(A6 ,[-1 ,8]), \
                                    torch.reshape(A7 ,[-1 ,8]) ,torch.reshape(A8 ,[-1 ,8])] ,axis=1)
        A_mat = a.permute(0 ,2 ,1) # BATCH_SIZE x 8 (A_i) x 8
        # Form b matrix
        b_mat = torch.matmul(self.Mb_tile, pred_pt4)

        H_8el,Lu=torch.solve(b_mat, A_mat)

        # Add ones to the last cols to reconstruct H for computing reprojection error
        H_9el = torch.cat([H_8el ,self.h_ones] ,1)
        H_flat = torch.reshape(H_9el, [-1 ,9])
        H_mat = torch.reshape(H_flat ,[-1 ,3 ,3])   # BATCH_SIZE x 3 x 3
        return H_mat











