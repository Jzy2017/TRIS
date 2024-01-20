import os
import cv2
import numpy as np
input1=os.listdir('../training_SR/input1')
input2=os.listdir('../training_SR/input1')
warp1=os.listdir('../training_SR/warp1')
label2=os.listdir('../training_SR/label2')
print(len(input1))
print(len(warp1))
print(len(label2))
label2.sort()
input2.sort()
input1.sort()
for i in range(49920):
    # if input1[i]!=warp1[i]:
    #     print(input1[i])
    #     print(warp1[i])
    #     print(label2[i])
    #     break
    input1_img=cv2.imread('../training_SR/input1/'+input1[i])
    input2_img=cv2.imread('../training_SR/input2/'+input2[i])
    label2_img=cv2.imread('../training_SR/label2/'+label2[i])
    cv2.imwrite('../training_SR/input1_new/'+str(i+1).zfill(6)+'.jpg',input1_img)
    cv2.imwrite('../training_SR/input2_new/'+str(i+1).zfill(6)+'.jpg',input2_img)
    cv2.imwrite('../training_SR/label2_new/'+str(i+1).zfill(6)+'.jpg',label2_img)
