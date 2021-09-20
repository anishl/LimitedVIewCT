import numpy as np
from numpy.lib import stride_tricks
import os
import itertools

def ext3Dpatch(data, blck, strd):
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6#.reshape(-1, *blck)

def load_walnutdata_raw(path,fname):
    return np.fromfile(os.path.join(path,fname),
                       dtype='float64').astype('float32')


def load_test_data(wal_num=3,lvl_num=1,
                   lenz=352,lenx=296,leny=400,frac=0.5,blck=[32,32,32],strd=[1,1,1],half='top',depth_z=31):
    flg=0
    if lvl_num>1:
        flg=1
        
    if lvl_num==1:
        lvl_num = 'ep'
    else:
        lvl_num = str(lvl_num-1)
    
    path='/n/escanaba/w/anishl/WalnutsOwnReconstructions/Walnut'+ str(wal_num)
    fname_tru='walnut'+ str(wal_num) + '.raw'
    fname_lv='walnut'+ str(wal_num) + '_lv4.1_'+ lvl_num + '.raw'
    
    # data_tru = load_walnutdata_raw(path,fname_tru).reshape([lenz,lenx,leny]).transpose(2,1,0)
    # data_lv = load_walnutdata_raw(path,fname_lv).reshape([lenz,lenx,leny]).transpose(2,1,0)
    
    data_tru = load_walnutdata_raw(path,fname_tru).reshape([lenz,lenx,leny])
    data_lv = load_walnutdata_raw(path,fname_lv).reshape([lenz,lenx,leny])

    if flg==1:
        data_lv=data_lv.transpose(2,1,0)
        # data_tru=data_tru.transpose(2,1,0)

    data_lv_copy = data_lv.copy()
        
    if half == 'top': 
        sub_blocks_tru = ext3Dpatch(data_tru[:int(np.floor(0.5*lenz)),:,:],blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv[:int(np.floor(0.5*lenz)),:,:],blck,strd)
    elif half=='full':
        sub_blocks_tru = ext3Dpatch(data_tru,blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv,blck,strd)
    else:
        start_z = int(np.floor(0.5*lenz))
        end_z = start_z+depth_z
        sub_blocks_tru = ext3Dpatch(data_tru[start_z:end_z,:,:],blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv[start_z:end_z,:,:],blck,strd)
        
    ixs = np.array([(i,j,k) 
                    for i,j,k in 
                    list(itertools.product(
                        range(sub_blocks_tru.shape[0]),range(sub_blocks_tru.shape[1]),range(sub_blocks_tru.shape[2])))])
    
    # For one volume only
    # ixs = np.array([(i,j) 
    #                 for i,j in 
    #                 list(itertools.product(
    #                     range(sub_blocks_tru.shape[1]),range(sub_blocks_tru.shape[2])))])
    
    # ixs = np.random.permutation(ixs)
    # ndata = ixs.shape[0]
    # ntrain = np.floor((1-frac)*ixs.shape[0]).astype('int32')
    if half == 'top':    
        return data_lv_copy[:int(np.floor(0.5*lenz)),:,:],data_tru[:int(np.floor(0.5*lenz)),:,:],sub_blocks_lv,ixs
    elif half == 'full':
        return data_lv_copy,data_tru,sub_blocks_lv,ixs
    else:
        return data_lv_copy[start_z:end_z,:,:],data_tru[start_z:end_z,:,:],sub_blocks_lv,ixs

if __name__ == "__main__":
    lv,tru,lv_blk,ixs = load_test_data(half='bot')
        
# TODO: get subblocks, get batches, check consistency between LV and FV patches

def load_small_batch_data(nData=500000,wal_num=1,lvl_num=1,
                    lenz=352,lenx=296,leny=400,blck=[8,500,500],strd=[1,1,1],half='top'):
    flg=0
    if lvl_num>1:
        flg=1
        
    if lvl_num==1:
        lvl_num = 'ep'
    else:
        lvl_num = str(lvl_num-1)
        
        
    
    path='/n/escanaba/w/anishl/WalnutsOwnReconstructions/Walnut'+str(wal_num)
    fname_tru='walnut'+str(wal_num)+'.raw'
    fname_lv='walnut'+str(wal_num)+'_lv4.1_'+lvl_num+'.raw'
    
    
    data_tru = load_walnutdata_raw(path,fname_tru).reshape([lenz,lenx,leny])
    data_lv = load_walnutdata_raw(path,fname_lv).reshape([lenz,lenx,leny])
    
    if flg==1:
        data_lv=data_lv.transpose(2,1,0)

    if half == 'top':
        print('only top half used for training')    
        sub_blocks_tru = ext3Dpatch(data_tru[:int(np.floor(0.5*lenz)),:,:],blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv[:int(np.floor(0.5*lenz)),:,:],blck,strd)
    elif half=='full':
        sub_blocks_tru = ext3Dpatch(data_tru,blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv,blck,strd)
    else:
        start_z = np.floor(0.5*lenz)
        end_z = start_z+blck[0]+1
        sub_blocks_tru = ext3Dpatch(data_tru[start_z:end_z,:,:],blck,strd)
        sub_blocks_lv = ext3Dpatch(data_lv[start_z:end_z,:,:],blck,strd)

    ixs0=np.random.randint(0,sub_blocks_tru.shape[0],[nData,1])
    ixs1=np.random.randint(0,sub_blocks_tru.shape[1],[nData,1])
    ixs2=np.random.randint(0,sub_blocks_tru.shape[2],[nData,1])



    ixs = np.concatenate((ixs0,ixs1,ixs2),axis=1)

    return sub_blocks_tru,sub_blocks_lv,ixs



