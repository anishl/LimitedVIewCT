#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 11:04:00 2021

@author: anishl
"""
import torch
import torch.nn

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        mse_loss = torch.nn.MSELoss(reduction='none')
        mse=mse_loss(input,target)
        
        mse = (mse * mask.float()).sum() # gives \sigma_euclidean over unmasked elements

        non_zero_elements = mask.sum()
        loss = mse / non_zero_elements
        
        return loss
    
class WalnutBounds(torch.nn.Module):
    def __init__(self):
        super(WalnutBounds,self).__init__()
        
    def return_mask(size,bounds):
        mask = torch.zeros(size)
        mask[bounds[4]:bounds[5],bounds[2]:bounds[3],bounds[0]:bounds[1]]=1
        return mask
        
    def getBounds(wal_num):
        if wal_num == 101:
            bounds=[45,474,70,430,50,475]
        if wal_num == 103:
            bounds=[100,420,95,412,60,460]
        if wal_num == 102:
            bounds=[95,425,80,400,60,445]
        if wal_num == 104:
            bounds=[55,420,80,450,40,480]
        if wal_num == 105:
            bounds=[45,405,75,425,40,460]
        if wal_num == 106:
            bounds=[40,430,55,455,30,470]
        return bounds