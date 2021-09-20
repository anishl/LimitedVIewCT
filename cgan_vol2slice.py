import argparse
import os
import numpy as np
import math

import sys
sys.path.insert(0, 'helpercode/')
from pytorch3dunet.unet3d import model as unet3dmod
from classifier3dcnn.model import ConvColumn5,ConvColumn3,ConvColumn7,ConvColumn9,ConvColumn2d
from vol2slicenet import vol2slice_blk
import matplotlib.pyplot as plt
from dbgfiles import dbgplot_v2s
import datautils
import lossutils as lu
from lossutils import WalnutBounds as wb

#import torchvision.transforms as transforms
#from torchvision.utils import save_image

# from torch.utils.data import DataLoader
# from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
# import torch.nn.functional as F
import torch
import pkbar

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'
#%% Setup argument input

#os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=40, help="number of epochs of training")
parser.add_argument("--lam", type=float, default=10, help="balance between pixel and adversarial loss 0<lam<1")
parser.add_argument("--d_int", type=int, default=10, help="iterations between discriminator training")
parser.add_argument("--batch_size", type=int, default=6, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

#parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
#parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
#parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
#parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
#parser.add_argument("--channels", type=int, default=1, help="number of image channels")
#parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

walnut_num = 101
lvl_num = 4

SAVE_PATH = './saved_models/'
batch_size = opt.batch_size
MODEL_NAME = 'v2sgenmsk_lvl'+str(lvl_num)+'_4.1v_ep_8x500'
lam = opt.lam
disc_train_interval = opt.d_int

cuda = True if torch.cuda.is_available() else False


## Decide which device we want to run on
# device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")
device = torch.device('cuda:0')

# Mask for walnut
msk = wb.return_mask([501,501,501],wb.getBounds(walnut_num))
msk = msk[:500,:500,:500]
msk_slc = msk[:,:,250].to(device)
# Loss functions
# generator_loss = torch.nn.MSELoss()
generator_loss = lu.MaskedMSELoss()
adversarial_loss = torch.nn.MSELoss()

#%% Load Data 
print('loading train dataset')
# Configure data loader (nData = 500000)
sub_blk_data_true,sub_blk_data_lv,train_ixs = datautils.load_small_batch_data(nData=(1900//batch_size + 1)*batch_size,
                                                                              wal_num=walnut_num,lvl_num=lvl_num,
                                                                              lenz=501,lenx=501,leny=501,half='full')

print('loaded datasets')

blk_depth = sub_blk_data_true.shape[3]
#%% Initialize generator and discriminator


generator= unet3dmod.ResidualUNet3D(1, 1, final_sigmoid=False, f_maps=64, layer_order='gcr',
                                num_groups=8, num_levels=1, is_segmentation=False, conv_padding=0)

v2s = vol2slice_blk.vol2slice(blk_depth)

discriminator = ConvColumn2d(1)
# discriminator = ConvColumn3(1)

# Optimizers
optimizer_G = torch.optim.Adam(list(generator.parameters())+list(v2s.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


#%% To Cuda
if cuda:
    generator = nn.DataParallel(generator)
    v2s = nn.DataParallel(v2s)
    discriminator = nn.DataParallel(discriminator)
    generator=generator.to(device)
    v2s=v2s.to(device)
    discriminator=discriminator.to(device)
    
    generator_loss=generator_loss.to(device)
    adversarial_loss=adversarial_loss.to(device)


#%% Train etc
# ----------
#  Training
# ----------
fig,axs = plt.subplots(1,3)
for epoch in range(opt.n_epochs):
    print('Epoch: %d/%d' % (epoch + 1, opt.n_epochs))
    kbar = pkbar.Kbar(target=train_ixs.shape[0], width=8)
    
    for batch_start_ix in np.arange(0,train_ixs.shape[0],batch_size):
            
        
        tru_data = torch.from_numpy(sub_blk_data_true[train_ixs[batch_start_ix:batch_start_ix+batch_size,0],
                                train_ixs[batch_start_ix:batch_start_ix+batch_size,1],
                                train_ixs[batch_start_ix:batch_start_ix+batch_size,2],:,:,:]).type(torch.FloatTensor).unsqueeze(1)
        
        lv_data = torch.from_numpy(sub_blk_data_lv[train_ixs[batch_start_ix:batch_start_ix+batch_size,0],
                                train_ixs[batch_start_ix:batch_start_ix+batch_size,1],
                                train_ixs[batch_start_ix:batch_start_ix+batch_size,2],:,:,:]).type(torch.FloatTensor).unsqueeze(1)
        
        #if cuda:        
        tru_data=tru_data[:,:,blk_depth//2].to(device)
        lv_data=lv_data.to(device)
        
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)
        
        # # Configure input
        # real_imgs = Variable(imgs.type(FloatTensor))
        # labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # # Sample noise and labels as generator input
        # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        # gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        est_data0 = generator(lv_data)
        est_data = v2s(est_data0.squeeze(1))

        # Loss measures generator's ability to fool the discriminator
        d_pred = discriminator(est_data.detach())
        #print(d_pred)
        g_loss1 =  (1*generator_loss(est_data,tru_data,msk_slc)) 
        lam = 10**torch.floor(torch.log10(g_loss1))
        g_loss2 = (lam*adversarial_loss(d_pred, valid)) 
        g_loss = g_loss1+g_loss2
        
        # if epoch>=1:
        #     dbgplot_v2s.dbgplt(lv_data, est_data0.squeeze(1), est_data, axs)

        
        g_loss.backward()
        optimizer_G.step()
            
        if batch_start_ix%(disc_train_interval*batch_size)==0:
        
        # ---------------------
        #  Train Discriminator
        # ---------------------

            optimizer_D.zero_grad()
    
            # Loss for real images
            d_real_pred = discriminator(tru_data)
            d_real_loss = adversarial_loss(d_real_pred, valid)
    
            # Loss for fake images
            d_fake_pred = discriminator(est_data.detach())
            d_fake_loss = adversarial_loss(d_fake_pred, fake)
    
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
    
            d_loss.backward()
            optimizer_D.step()
            
        torch.cuda.empty_cache()
        
        kbar.update(batch_start_ix, values=[("Scaled D loss", g_loss2.item()), ("G loss", g_loss1.item())])
        
        # print(
        #     "[Epoch %d/%d] [Processed %d/%d] [D loss: %f] [G loss: %f]"
        #     % (epoch, opt.n_epochs, batch_start_ix, train_ixs.shape[0], d_loss.item(), g_loss.item())
        # )
        
    if epoch%1==0:
        torch.save(generator.state_dict(), SAVE_PATH+MODEL_NAME+'_gen.pth')        
        torch.save(v2s.state_dict(), SAVE_PATH+MODEL_NAME+'_v2s.pth')        

#%% extra functions as needed
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
