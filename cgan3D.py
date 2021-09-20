import argparse
import os
import numpy as np
import math

import sys
sys.path.insert(0, 'helpercode/')
from pytorch3dunet.unet3d import model as unet3dmod
from classifier3dcnn.model import ConvColumn5
import datautils

import torchvision.transforms as transforms
from torchvision.utils import save_image

# from torch.utils.data import DataLoader
# from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
# import torch.nn.functional as F
import torch
import pkbar

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
#%% Setup argument input

#os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--lam", type=float, default=0.85, help="balance between pixel and adversarial loss 0<lam<1")
parser.add_argument("--d_int", type=int, default=10, help="iterations between discriminator training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
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

SAVE_PATH = './saved_models/'
MODEL_NAME = 'generator_fullbatch'
batch_size = opt.batch_size
lam = opt.lam
disc_train_interval = opt.d_int

cuda = True if torch.cuda.is_available() else False


## Decide which device we want to run on
# device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")
device = torch.device('cuda:0')

# Loss functions
generator_loss = torch.nn.MSELoss()
adversarial_loss = torch.nn.MSELoss()
#%% Initialize generator and discriminator


generator= unet3dmod.ResidualUNet3D(1, 1, final_sigmoid=False, f_maps=64, layer_order='gcr',
                                num_groups=8, num_levels=3, is_segmentation=False, conv_padding=1)
discriminator = ConvColumn5(1)

#%% Load Dataset etc.
if cuda:
    generator = nn.DataParallel(generator,device_ids=[0,1,2])
    discriminator = nn.DataParallel(discriminator,device_ids=[0,1,2])
    generator=generator.to(device)
    discriminator=discriminator.to(device)
    
    generator_loss=generator_loss.to(device)
    adversarial_loss=adversarial_loss.to(device)

print('loading train dataset')
# Configure data loader
sub_blk_data_true,sub_blk_data_lv,train_ixs,test_ixs = datautils.load_batch_data()

print('loaded datasets')

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


#%% Train etc
# ----------
#  Training
# ----------

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
        tru_data=tru_data.to(device)
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
        est_data = generator(lv_data)

        # Loss measures generator's ability to fool the discriminator
        d_pred = discriminator(est_data.detach())
        #print(d_pred)
        g_loss1 =  (1*generator_loss(est_data,tru_data)) 
        lam = 10**torch.floor(torch.log10(g_loss1))
        g_loss2 = (lam*adversarial_loss(d_pred, valid)) 
        g_loss = g_loss1+g_loss2
        
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
            
        kbar.update(batch_start_ix, values=[("Scaled D loss", g_loss2.item()), ("G loss", g_loss1.item())])
        
        # print(
        #     "[Epoch %d/%d] [Processed %d/%d] [D loss: %f] [G loss: %f]"
        #     % (epoch, opt.n_epochs, batch_start_ix, train_ixs.shape[0], d_loss.item(), g_loss.item())
        # )
        
    #if epoch%1==0:
    torch.save(generator.state_dict(), SAVE_PATH+MODEL_NAME+'.pth')        

#%% extra functions as needed
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
