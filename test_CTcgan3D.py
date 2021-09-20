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
import datautils
import pkbar
import h5py
import scipy.io as sio

import sys
sys.path.insert(0, 'helpercode/')
from pytorch3dunet.unet3d import model as unet3dmod

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

#%% Specify model name and path
PATH = './saved_models/'
MODEL_NAME = 'generator_full_8v_ep'


#%% load model for testing in eval mode
device = torch.device('cuda:0')
model = nn.DataParallel(unet3dmod.ResidualUNet3D(1, 1, final_sigmoid=False, f_maps=64, layer_order='gcr',
                                num_groups=8, num_levels=3, is_segmentation=False, conv_padding=1))

# model = model.load_state_dict(torch.load(PATH+MODEL_NAME+'.pth',map_location='cpu'))
model.load_state_dict(torch.load(PATH+MODEL_NAME+'.pth'))
model = model.to(device)
model.eval()

#%% spcify parameters for testing
strd = [2,2,2]
mode = 'subvol'
batch_size=160
blck = [32,32,32]

#%% load data for test
data_lv,data_true,sub_blk_data_lv,train_ixs,walnut_num = datautils.load_test_data(half='full',blck=blck,strd=strd,
                                                                       lenz=501,lenx=501,leny=501,depth_z=501)

blk_size = sub_blk_data_lv.shape[3:]


data_denoised = np.zeros_like(data_true)
weights = np.zeros_like(data_true)
data_denoised_views = datautils.ext3Dpatch(data_denoised, blk_size, strd)
weights_views = datautils.ext3Dpatch(weights, blk_size, strd)

kbar = pkbar.Kbar(target=train_ixs.shape[0], width=20)
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
        # out_data = out_data.cpu().detach().numpy()
        #ix_z=0
        
        data_denoised_views[ix_z,ix_y,ix_x,:,:,:]+=out_data.cpu().detach().numpy()
        weights_views[ix_z,ix_y,ix_x,:,:,:]+=torch.ones_like(out_data).cpu().detach().numpy()
        kbar.update(batch_start_ix)
    
    
    data_out_normalized = data_denoised/weights
else:
    data_lv=torch.from_numpy(data_lv).unsqueeze(0).unsqueeze(0).to(device)
    out_data = model(data_lv)
    
    
sio.savemat('results/'+MODEL_NAME+str(blck)+'_'+walnut_num+'_60.mat',{"data_true":data_true,"data_out_normalized":data_out_normalized,"data_lv":data_lv})
#%% Visualization
loadData=False
# loadData=True

if not loadData:

    fig,axs = plt.subplots(1,3)
    slc_num = 200#18
    colmap = 'hot'
    
    data_true1 = data_true.transpose(2,1,0)
    
    ax = axs[0]
    im = ax.imshow(data_true1[slc_num],cmap=colmap, interpolation='none',vmax=50000*1e-6)
    ax.set_title('True');ax.axis('off')
    fig.colorbar(im,ax = ax,shrink = 0.25)
    
    ax = axs[1]
    im = ax.imshow(data_out_normalized[slc_num],cmap=colmap, interpolation='none',vmax=50000*1e-6)
    ax.set_title('Destreaked (Gen)');ax.axis('off')
    fig.colorbar(im,ax = ax,shrink = 0.25)
    
    
    ax = axs[2]
    im = ax.imshow(data_lv[slc_num],cmap=colmap, interpolation='none',vmax=50000*1e-6,vmin=0)
    ax.set_title('EP recon');ax.axis('off')
    fig.colorbar(im,ax = ax,shrink = 0.25)
    



if loadData:
    s = sio.loadmat('results/'+MODEL_NAME+str(blck)+'_'+walnut_num+'_60.mat')
    data_true=s["data_true"]
    data_out_normalized=s["data_out_normalized"]
    data_lv=s["data_lv"]


#%% Comparison
vis = False


# blk_num = 20 #20
if vis:
    fig,axs = plt.subplots(1,4)
    # fig,axs = plt.subplots(1,5)
    colmap = 'gray'
    ax = axs[0]
    im = ax.imshow(data_true[slc_num,:,:],cmap=colmap, interpolation='none',vmax=20000)
    ax.set_title('True');ax.axis('off')
    fig.colorbar(im,ax = ax,shrink = 0.25)
    
    ax = axs[1]
    im = ax.imshow(data_out_normalized_gen[slc_num,:,:],cmap=colmap, interpolation='none',vmax=20000)
    ax.set_title('Denoised (Gen)');ax.axis('off')
    fig.colorbar(im,ax = ax,shrink = 0.25)
    
    ax = axs[2]
    im = ax.imshow(data_out_normalized_sup[slc_num,:,:],cmap=colmap, interpolation='none',vmax=20000)
    ax.set_title('Denoised (Sup)');ax.axis('off')
    fig.colorbar(im,ax = ax,shrink = 0.25)
    
    # ax = axs[2]
    # im = ax.imshow((np.abs(data_out_normalized-data_true))[slc_num,:,:],cmap=colmap,interpolation='none',vmax=1000)
    # ax.set_title('abs Error');ax.axis('off')
    # fig.colorbar(im,ax = ax,shrink = 0.25)
    
    ax = axs[3]
    im = ax.imshow(data_lv[slc_num,:,:],cmap=colmap, interpolation='none',vmax=20000)
    ax.set_title('FDK');ax.axis('off')
    fig.colorbar(im,ax = ax,shrink = 0.25)

        
    
    
    # torch.cuda.empty_cache()

#%% Comparison GAN vs pixelwise loss
if False:


    c = h5py.File('results/test-2020-07-06.mat')
    data_tv = c['/xos_good'][:,:,:]
    data_tv = data_tv[int(0.5*data_tv.shape[0]):int(0.5*data_tv.shape[0])+32]
    c = np.load('results/2020-07-04-small-batch-gen.npz')
    data_out_normalized_gen = c['data_out_normalized']
    c = np.load('results/2020-07-04-small-batch-sup.npz')
    data_out_normalized_sup = c['data_out_normalized']
    data_true = c['data_true']
    data_lv = c['data_lv']

    slc_num = 16
    colmap = 'hot'    
    
    fig,axs = plt.subplots(4,2)
    
    ax = axs[0,0]
    im = ax.imshow(data_true[slc_num,:,:],cmap=colmap, interpolation='none',vmax=2e4)
    ax.set_title('Ground Truth');ax.axis('off')
    cb = fig.colorbar(im,ax = ax,shrink = 2*0.5, format = '%0.0e')
    cb.ax.locator_params(nbins=1)
    
    ax = axs[0,1]
    im = ax.imshow(data_lv[slc_num,:,:],cmap=colmap, interpolation='none',vmax=2e4)
    ax.set_title('FDK (#views=16)');ax.axis('off')
    cb = fig.colorbar(im,ax = ax,shrink = 2*0.5, format = '%0.0e')
    cb.ax.locator_params(nbins=1)

    
    ax = axs[1,0]
    im = ax.imshow(data_out_normalized_gen[slc_num,:,:],cmap=colmap, interpolation='none',vmax=2e4)
    ax.set_title('Destreaked (Supervised+Adversarial Loss)');ax.axis('off')
    cb = fig.colorbar(im,ax = ax,shrink = 2*0.5, format = '%0.0e')
    cb.ax.locator_params(nbins=1)
    
    ax = axs[1,1]
    im = ax.imshow((np.abs(data_out_normalized_gen-data_true))[slc_num,:,:],cmap=colmap,interpolation='none',vmax=5e3)
    ax.set_title('abs Error (Sup+Adv) \ MAE: %.1f' %(np.abs(data_out_normalized_gen-data_true))[slc_num,:,:].mean());ax.axis('off')
    cb = fig.colorbar(im,ax = ax,shrink = 2*0.5, format = '%0.0e')
    cb.ax.locator_params(nbins=1)
    
    ax = axs[2,0]
    im = ax.imshow(data_out_normalized_sup[slc_num,:,:],cmap=colmap, interpolation='none',vmax=2e4)
    ax.set_title('Destreaked (Supervised CNN) ', );ax.axis('off')
    cb = fig.colorbar(im,ax = ax,shrink = 2*0.5, format = '%0.0e')
    cb.ax.locator_params(nbins=1)
    
    
    ax = axs[2,1]
    im = ax.imshow((np.abs(data_out_normalized_sup-data_true))[slc_num,:,:],cmap=colmap,interpolation='none',vmax=5e3)
    ax.set_title('abs Error (Supervised CNN) \ MAE: %.1f' %(np.abs(data_out_normalized_sup-data_true))[slc_num,:,:].mean());ax.axis('off')
    cb = fig.colorbar(im,ax = ax,shrink = 2*0.5, format = '%0.0e')
    cb.ax.locator_params(nbins=1)
    plt.tight_layout(pad=0.1, w_pad=0, h_pad=0.1)
    
    ax = axs[3,0]
    im = ax.imshow(data_tv[slc_num,:,:],cmap=colmap, interpolation='none',vmax=2e4)
    ax.set_title('Model-Based Image Recon (PWLS-OS) ', );ax.axis('off')
    cb = fig.colorbar(im,ax = ax,shrink = 2*0.5, format = '%0.0e')
    cb.ax.locator_params(nbins=1)
    
    
    ax = axs[3,1]
    im = ax.imshow((np.abs(data_tv-data_true))[slc_num,:,:],cmap=colmap,interpolation='none',vmax=5e3)
    ax.set_title('abs Error (PWLS-OS) \ MAE: %.1f' %(np.abs(data_tv-data_true))[slc_num,:,:].mean());ax.axis('off')
    cb = fig.colorbar(im,ax = ax,shrink = 2*0.5, format = '%0.0e')
    cb.ax.locator_params(nbins=1)
    plt.tight_layout(pad=0.1, w_pad=0, h_pad=0.1)


"""
top=0.955,
bottom=0.02,
left=0.01,
right=0.55,
hspace=0.1,
wspace=0.0
"""