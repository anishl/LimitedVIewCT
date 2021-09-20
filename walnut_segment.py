import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import scipy.io as sio

# filename = 'Reconstructions/full_AGD_50_000{:03d}.tiff'
# walnut = np.stack([Image.open(filename.format(i)) for i in range(501)])
#%% Specify model name
MODEL_NAME = 'generator_full_8v_ep'
blck = [32,32,32]

s = sio.loadmat('results/'+MODEL_NAME+str(blck)+'.mat')
data_true=s["data_true"]
data_out_normalized=s["data_out_normalized"]
data_lv=s["data_lv"]

walnut = data_out_normalized
walnut_true = data_true

def get_bins(data):
    #takes in an array of attenuation coefficients (a reconstruction), and tries to return reasonable bins for different materaial attenuation coefficients for classification,
    #returns a list of the bin idices, and a list of the maximum peaks of the histogram
    hist_data = plt.hist(data.ravel(),bins=200)
    maxs = list(argrelextrema(hist_data[0],np.greater,order=3, mode='wrap')[0])
    maxs = [a for a in maxs if hist_data[0][a] > 1e3] #only want real local maxima

    maxs_densities = [hist_data[1][a] for a in maxs]
    # spaced mins
    midpoint_bins = [-np.inf] + [0.5*(maxs_densities[i]+maxs_densities[i+1]) for i in range(len(maxs_densities)-1)] + [np.inf]
    #min_bins = [-np.inf] + [0.5*(maxs[i]+maxs[i+1]) for i in range(len(maxs)-1)] + [np.inf]
    bins = midpoint_bins
    return bins, list(map(lambda a: 0.5*(hist_data[1][a]+hist_data[1][a+1]), maxs))

def classify(data,bins):
    #returns an array of diminsions of data x len(bins)-1, the result[...,i] array will contain all the values of the data that belong in the ith bin.
    result = np.stack([np.where(np.logical_and(data>=bins[i], data<bins[i+1]), data, np.nan) for i in range(len(bins)-1)],axis=-1)
    return result
        

fig, ax = plt.subplots()
bins, maxs =  get_bins(walnut)
bins_true, maxs_true = get_bins(walnut_true)

for i,a in enumerate(maxs):
    plt.axvline(x=a, c='k', linestyle=':', linewidth=1)
plt.yscale('log')
plt.title('walnut 2 reconstruction densities')
plt.xlabel('density')
ax.xaxis.set_label_coords(.5, -0.15)
plt.ylabel('count')
fig.show()
plt.savefig('density_histogram.png',bbox_inches='tight',dpi=200)

out = classify(walnut_true,bins_true)
out = out[:,:,:,1:]

fig = plt.figure()
plt.imshow(out[:,:,250,:]/np.nanmax(out[:,:,250,:]))
fig.show()
# plt.savefig('view0.png',dpi=200,bbox_inches='tight')

fig = plt.figure()
plt.imshow(out[:,250,:,:]/np.nanmax(out[:,250,:,:]))
fig.show()
# plt.savefig('view1.png',dpi=200,bbox_inches='tight')

fig = plt.figure()
plt.imshow(out[250,:,:,:]/np.nanmax(out[250,:,:,:]))
fig.show()
# plt.savefig('view2.png',dpi=200,bbox_inches='tight')


    
