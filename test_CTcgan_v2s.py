#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:13:04 2020

@author: anishl
"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import time
import datautils
import pkbar
import h5py
import scipy.io as sio
from lossutils import WalnutBounds as wb

import sys
sys.path.insert(0, 'helpercode/')
from pytorch3dunet.unet3d import model as unet3dmod
from vol2slicenet import vol2slice_blk
from dbgfiles import dbgplot_v2s

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

#%% Specify model name, test data and path
lvl_num = 4
walnut_num = 104

PATH = './saved_models/'
MODEL_NAME = 'v2sgenmsk_lvl'+ str(lvl_num)+'_4.1v_ep_8x500'


#%% load data for test
strd = [1,1,1]
mode = 'subvol'
batch_size= 3
blck=[8,500,500]

data_lv,data_true,sub_blk_data_lv,train_ixs = datautils.load_test_data(wal_num=walnut_num,lvl_num=lvl_num,half='full',blck=blck,strd=strd,
                                                                       lenz=501,lenx=501,leny=501,depth_z=64)

blk_size = sub_blk_data_lv.shape[3:]

blk_depth = blk_size[0]

#%% load model for testing in eval mode
torch.cuda.empty_cache()
device = torch.device('cuda:0')
# device1 = torch.device('cuda:1')
model = nn.DataParallel(unet3dmod.ResidualUNet3D(1, 1, final_sigmoid=False, f_maps=64, layer_order='gcr',
                                num_groups=8, num_levels=1, is_segmentation=False, conv_padding=0))

# model = model.load_state_dict(torch.load(PATH+MODEL_NAME+'.pth',map_location='cpu'))
model.load_state_dict(torch.load(PATH+MODEL_NAME+'_gen.pth'))
model = model.to(device)
model = model.eval()


v2s = nn.DataParallel(vol2slice_blk.vol2slice(blk_depth))
v2s.load_state_dict(torch.load(PATH+MODEL_NAME+'_v2s.pth'))
v2s = v2s.to(device)
v2s = v2s.eval()

msk = wb.return_mask([501,501,501],wb.getBounds(walnut_num))
msk = msk[:500,:500,:500]
msk_slc = msk[:,:,250].to(device)

#%%

data_denoised = np.zeros_like(data_true)
weights = np.zeros_like(data_true)
data_denoised_views = datautils.ext3Dpatch(data_denoised, blk_size, strd)
weights_views = datautils.ext3Dpatch(weights, blk_size, strd)

kbar = pkbar.Kbar(target=train_ixs.shape[0], width=20)
fig,axs = plt.subplots(1,3)

if mode == 'subvol':
    for batch_start_ix in np.arange(0,train_ixs.shape[0],batch_size):
                
        
        ix_z = train_ixs[batch_start_ix:batch_start_ix+batch_size,0]
        ix_x = train_ixs[batch_start_ix:batch_start_ix+batch_size,2]
        ix_y = train_ixs[batch_start_ix:batch_start_ix+batch_size,1]
        
        lv_data = torch.from_numpy(sub_blk_data_lv[ix_z,
                                ix_y,ix_x,:,:,:]
                                   ).type(torch.FloatTensor).unsqueeze(1)
        lv_data = lv_data.to(device)
        # pass data thru denoiser
        out_data = model(lv_data).squeeze(1)
        out_data2 = v2s(out_data)
        out_data2 = out_data2*msk_slc.float()
        # out_data = out_data.cpu().detach().numpy()
        #ix_z=0
        
        # PLOT FOR Debug
        # dbgplot_v2s.dbgplt(lv_data, out_data, out_data2, axs)
        
        if batch_start_ix>=1000:
            ctr=1
        
        data_denoised_views[ix_z,ix_y,ix_x,blk_depth//2,:,:]+=out_data2.cpu().detach().numpy().squeeze(1)
        weights_views[ix_z,ix_y,ix_x,blk_depth//2,:,:]+=torch.ones_like(out_data2).cpu().detach().numpy().squeeze(1)
        kbar.update(batch_start_ix)
    
    
    data_out_normalized = data_denoised/weights
else:
    data_lv=torch.from_numpy(data_lv).unsqueeze(0).unsqueeze(0).to(device)
    out_data = model(data_lv)
    
    
sio.savemat('results/'+MODEL_NAME+str(blck)+'_'+str(walnut_num)+'.1.mat',{"data_true":data_true,"data_out_normalized":data_out_normalized,"data_lv":data_lv})
#%% Visualization
loadData=False
# loadData=True

if not loadData:

    fig,axs = plt.subplots(1,3)
    slc_num = 200#18
    colmap = 'hot'
    
    # data_true1 = data_true.transpose(2,1,0)
    
    ax = axs[0]
    im = ax.imshow(data_true[:,slc_num],cmap=colmap, interpolation='none',vmax=50000*1e-6,vmin=0)
    ax.set_title('True');ax.axis('off')
    fig.colorbar(im,ax = ax,shrink = 0.25)
    
    ax = axs[1]
    im = ax.imshow(data_out_normalized[:,slc_num],cmap=colmap, interpolation='none',vmax=50000*1e-6,vmin=0)
    ax.set_title('Destreaked (Gen)');ax.axis('off')
    fig.colorbar(im,ax = ax,shrink = 0.25)
    
    
    ax = axs[2]
    im = ax.imshow(data_lv[:,slc_num],cmap=colmap, interpolation='none',vmax=50000*1e-6,vmin=0)
    ax.set_title('EP recon');ax.axis('off')
    fig.colorbar(im,ax = ax,shrink = 0.25)
    



if loadData:
    s = sio.loadmat('results/'+MODEL_NAME+str(blck)+'_'+walnut_num+'.mat')
    data_true=s["data_true"]
    data_out_normalized=s["data_out_normalized"]
    data_lv=s["data_lv"]







