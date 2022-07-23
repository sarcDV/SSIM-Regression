import os, sys
import random

# --------
import torchvision.utils as vutils
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# --------
import nibabel as nib
from skimage.transform import resize
from models.ReconResNetV2 import ResNet


def main():
    """run with the following command:
       python SSIM-regression.py nii-file (file.nii.gz)
    """
    clinical_evaluate_nii(sys.argv[1]) #,sys.argv[2],sys.argv[3])
    return

###########################################################################
##### Auxiliaries  ########################################################
###########################################################################
def padding(array, xx, yy):
    h = array.shape[0]
    w = array.shape[1]
    a = (xx - h) // 2
    aa = xx - a - h
    b = (yy - w) // 2
    bb = yy - b - w
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


def PadResize(imgin, finalsize):
    ## find biggest size:
    dim_ = imgin.shape
    maxElement = np.where(dim_== np.amax(dim_))
    imgout = padding(imgin, dim_[maxElement[0][0]], dim_[maxElement[0][0]])
    imgres = resize(imgout, (finalsize, finalsize), anti_aliasing=True )
    return imgres # imgout, imgres
###########################################################################
##### Cut noise level  ####################################################
###########################################################################

def cutNoise(img, level):
    adjimg = (img>level)*1.0*img
    ## normalize again:
    adjimg = (adjimg-adjimg.min())/(adjimg.max()-adjimg.min())

    return adjimg # np.abs(adjimg)+1e-16
###########################################################################

def clinical_evaluate_nii(filein, chkin='./Weights/RN18.pth.tar', neuralmodel='RN18'):
    ### ----------------------------------------------------- ###
    device = torch.device("cuda:0") 
    checkpoint = chkin
    batch_size_, patches, channels, size_= 1,1,1,256
    level_noise = 0.025
    ### ----------------------------------------------------- ###
    if neuralmodel == 'RN18':
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_classes = 1
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        model.to(device)
    else:
        model = models.resnet101(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_classes = 1
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        model.to(device)
    
    chk = torch.load(checkpoint, map_location=device)
    model.load_state_dict(chk['state_dict'] )
    model.eval()
    ### ----------------------------------------------------- ###
    ### ----------------------------------------------------- ###
    orig_stdout = sys.stdout
    f = open(str(filein[0:len(filein)-7])+'_report.txt', 'w')
    sys.stdout = f
    ### ----------------------------------------------------- ###
    ### ----------------------------------------------------- ###
    
    a = nib.load(filein).get_fdata()
    ahead = nib.load(filein).header
    print(str(filein))
    print("\n","#### HEADER #####\n\n",ahead,"\n")
    ## original resolution & size ###
    print("#### Size #####\n\n",ahead['dim'][1:4],"\n")
    print("#### Resolution #####\n\n",ahead['pixdim'][1:4],"\n")
    if len(a.shape)>3:
        a = a[:,:,:,0]
    
    SSIMarray = np.zeros((a.shape[2]))
    print("#### Slice, SSIM ####")
    with torch.no_grad():
        for ii in range(0, a.shape[2]):
            img = cutNoise(a[:,:,ii], level_noise)
            img = PadResize(img, size_)
            img = torch.unsqueeze(torch.tensor(img).to(device),1)
            img = torch.reshape(img, (batch_size_*patches, channels, size_,size_))
            pred = model(img.float())
            print("Slice: "+str(ii+1)+", "+str(pred.detach().cpu().numpy()[0][0]))
            SSIMarray[ii] = pred.detach().cpu().numpy()[0][0]
    
    print("\n")
    print("Maximum: ", str(np.max(SSIMarray)))
    print("Minimum: ", str(np.min(SSIMarray)))
    print("Mean value: ", str(np.mean(SSIMarray)))
    print("Standard dev.: ", str(np.std(SSIMarray)))
    print("\n")
    ### ----------------------------------------------------- ###
    ### ----------------------------------------------------- ###
    sys.stdout = orig_stdout
    f.close()
if __name__ == "__main__":
	main()
