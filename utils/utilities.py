import os, sys, glob
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import nibabel as nib
import random
from numpy.lib.arraypad import pad
import torch
from statistics import median
# from scipy.misc import imrotate 
## scipy.ndimage.interpolation.rotate 
from scipy import ndimage, misc
from skimage import exposure
from progressbar import ProgressBar
# from skimage.metrics import structural_similarity as ssim
from skimage.metrics import (normalized_root_mse, peak_signal_noise_ratio,
                             structural_similarity)
import torchvision.utils as vutils
import torchio
import time
import multiprocessing.dummy as multiprocessing
from tqdm import tqdm
###########################################################################
##### Auxiliaries  ########################################################
###########################################################################

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
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
##### Contrast Augmentation  ##############################################
###########################################################################

def randomContrastAug(img):
    expo_selection = np.random.randint(0,5,1)
    if expo_selection[0] == 0:
        adjimg = exposure.adjust_gamma(img, np.random.uniform(0.75, 1.75, 1)[0])
    elif expo_selection[0] == 1:
        adjimg = exposure.equalize_adapthist(img, 
											kernel_size=int(np.random.randint(25, high=100, size=(1))[0]), #21, 
											clip_limit=0.01, 
											nbins=512)
    elif expo_selection[0] == 2:
        adjimg = exposure.adjust_sigmoid(img, 
	 								   cutoff=np.random.uniform(0.01, 0.75, 1)[0], #0.5, 
	  								   gain=int(np.random.randint(1, high=4, size=(1))[0]), #10, 
	  								   inv=False)
    elif expo_selection[0] == 3:
       adjimg = np.abs(exposure.adjust_log(img, np.random.uniform(-0.5, 0.5, 1)[0]))
    else:
        adjimg = img

    ## normalize again:
    adjimg = (adjimg-adjimg.min())/(adjimg.max()+1e-16-adjimg.min())

    return adjimg #, expo_selection[0]

###########################################################################
##### Cut noise level  ####################################################
###########################################################################

def cutNoise(img, level):
    adjimg = (img>level)*1.0*img
    ## normalize again:
    adjimg = (adjimg-adjimg.min())/(adjimg.max()+1e-16-adjimg.min())

    return adjimg 

###########################################################################
##### Motion  #############################################################
###########################################################################

class Motion2DOld():
    def __init__(self, sigma_range=(0.10, 2.5), n_threads=10):
        self.sigma_range = sigma_range
        self.n_threads = n_threads

    def __perform_singlePE(self, idx):
        rot = self.sigma*random.randint(-1,1)
        img_aux = ndimage.rotate(self.img, rot, reshape=False)
        # rot = np.random.uniform(self.mu, self.sigma, 1)*random.randint(-1,1)
        # rot = np.random.normal(self.mu, self.sigma, 1)*random.randint(-1,1)
        # img_aux = ndimage.rotate(self.img, rot[0], reshape=False)
        img_h = np.fft.fft2(img_aux)
        if self.axis_selection == 0:
            self.aux[:,idx]=img_h[:,idx]
        else:
            self.aux[idx,:]=img_h[idx,:]

    def __call__(self, img):
        self.img = img
        self.aux = np.zeros(img.shape) + 0j
        self.axis_selection = np.random.randint(0,2,1)[0]
        self.mu=0
        self.sigma=np.random.uniform(self.sigma_range[0], self.sigma_range[1], 1)[0]
        if self.n_threads > 1:
            pool = multiprocessing.Pool(self.n_threads)
            pool.map(self.__perform_singlePE, range(self.aux.shape[1] if self.axis_selection == 0 else self.aux.shape[0]))
        else:
            for idx in range(self.aux.shape[1] if self.axis_selection == 0 else self.aux.shape[0]):
                self.__perform_singlePE(idx)
        cor =np.abs(np.fft.ifft2(self.aux)) 
        del self.img, self.aux, self.axis_selection, self.mu, self.sigma
        return cor/(cor.max()+1e-16)

class Motion2D():
    def __init__(self, sigma_range=(0.10, 2.5), restore_original=5e-2, n_threads=10):
        self.sigma_range = sigma_range
        self.restore_original = restore_original
        self.n_threads = n_threads

    def __perform_singlePE(self, idx):
        img_aux = ndimage.rotate(self.img, self.random_rots[idx], reshape=False)
        img_h = np.fft.fft2(img_aux)            
        if self.axis_selection == 0:
            self.aux[:,self.portion[idx]]=img_h[:,self.portion[idx]]  
        else:
            self.aux[self.portion[idx],:]=img_h[self.portion[idx],:]  

    def __call__(self, img):
        self.img = img
        self.aux = np.zeros(img.shape) + 0j
        self.axis_selection = np.random.randint(0,2,1)[0]

        if self.axis_selection == 0:
            dim = 1
        else:
            dim = 0

        n_ = np.random.randint(2,8,1)[0]
        intext_ = np.random.randint(0,2,1)[0]
        if intext_ == 0:
            portiona = np.sort(np.unique(np.random.randint(low=0, 
                                                        high=int(img.shape[dim]//n_), 
                                                        size=int(img.shape[dim]//2*n_), dtype=int)))
            portionb = np.sort(np.unique(np.random.randint(low=int((n_-1)*img.shape[dim]//n_), 
                                                        high=int(img.shape[dim]), 
                                                        size=int(img.shape[dim]//2*n_), dtype=int))) 
            self.portion = np.concatenate((portiona, portionb))  
        else:
            self.portion = np.sort(np.unique(np.random.randint(low=int(img.shape[dim]//2)-int(img.shape[dim]//n_+1), 
                                                     high=int(img.shape[dim]//2)+int(img.shape[dim]//n_+1), 
                                                     size=int(img.shape[dim]//n_+1), dtype=int)))
        self.sigma=np.random.uniform(self.sigma_range[0], self.sigma_range[1], 1)[0]
        self.random_rots = self.sigma * np.random.randint(-1,1,len(self.portion))
        #  self.random_rots = np.random.randint(-4,4,len(self.portion))

        if self.n_threads > 1:
            pool = multiprocessing.Pool(self.n_threads)
            pool.map(self.__perform_singlePE, range(len(self.portion)-1))
        else:
            for idx in range(len(self.portion)-1):
                self.__perform_singlePE(idx)     
        cor =np.abs(np.fft.ifft2(self.aux)) # + self.restore_original *img

        del self.img, self.aux, self.axis_selection, self.portion, self.random_rots
        return cor/(cor.max()+1e-16)
###########################################################################
##### slice selection  ####################################################
###########################################################################
def select_slice(test):
    rndslice_ = np.random.randint(low=int(0),high=int(test.shape[2]), size=1)
    img = (test[:,:,rndslice_[0]])
    return img/(img.max()+1e-16)

def select_slice_orientation(test, orientation):
    if orientation == 3:
        if test.shape[2] > (test.shape[0]//2):
            rnd_orient = np.random.randint(0,3,1)[0]
            # print(rnd_orient)
            if rnd_orient == 0:
                rndslice_ = np.random.randint(low=int(test.shape[1]//2)-int(test.shape[1]//4), 
                                            high=int(test.shape[1]//2)+int(test.shape[1]//4), 
                                            size=1)
                
                img = (test[:,rndslice_[0],:])
            elif rnd_orient == 1:
                rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                            high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                            size=1)
                                        
                img = np.rot90(test[:,:,rndslice_[0]])    
            else:
                rndslice_ = np.random.randint(low=int(test.shape[0]//2)-int(test.shape[0]//4), 
                                            high=int(test.shape[0]//2)+int(test.shape[0]//4), 
                                            size=1)                            
                img = np.flipud(test[rndslice_[0],:,:])
        else:
            rnd_orient = 1
            rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                            high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                            size=1)                           
            img = np.rot90(test[:,:,rndslice_[0]]) 
    
    elif orientation == 0:
        rnd_orient = 0 
        rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                            high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                            size=1)
        img = np.rot90(test[:,:,rndslice_[0]]) 
    elif orientation == 1:
        rnd_orient = 1
        rndslice_ = np.random.randint(low=int(test.shape[1]//2)-int(test.shape[1]//4), 
                                            high=int(test.shape[1]//2)+int(test.shape[1]//4), 
                                            size=1)       
        img = (test[:,rndslice_[0],:])
    elif orientation == 2: 
        rnd_orient = 2
        rndslice_ = np.random.randint(low=int(test.shape[0]//2)-int(test.shape[0]//4), 
                                            high=int(test.shape[0]//2)+int(test.shape[0]//4), 
                                            size=1)                            
        img = np.flipud(test[rndslice_[0],:,:])

    img = (img-img.min())/(img.max()+1e-16-img.min())
            
    return img, rndslice_, rnd_orient

###########################################################################
##### Motion Corruption Class  ############################################
###########################################################################
class MoCoDatasetRegressionUpdated():
    """Motion Correction Dataset"""
    def __init__(self, input_list, 
                       patches=10, 
                       size=256,
                       modalityMotion=2, # 0 only reality motion, 1 only TorchIO, 2 combined reality+TIO
                       sigma_range=(0.0, 3.0),
                       level_noise=0.025,
                       num_ghosts=5,
                       axes=2,
                       intensity=0.75,
                       restore=0.02,
                       degrees=10,
                       translation=10,
                       num_transforms=10,
                       image_interpolation='linear',
                       transform=None):
        """
        Args:
            input list (string): Path to the list of files;
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files_in_ = list(input_list)
        self.patches = patches
        self.size = size
        self.modalityMotion = modalityMotion
        self.transform = transform
        self.sigma_range = sigma_range
        self.level_noise = level_noise
        self.num_ghosts = num_ghosts
        self.axes = axes
        self.intensity = intensity
        self.restore = restore
        self.degrees = degrees
        self.translation = translation
        self.num_transforms = num_transforms
        self.image_interpolation = image_interpolation
        self.cter = Motion2DOld(n_threads=32, sigma_range=self.sigma_range)

    def __len__(self):
        return len(self.files_in_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_ = os.path.join(self.files_in_[idx])
        ## print(img_name_in_)
        image_in_ = nib.load(img_name_in_).get_fdata()
        
        size_ = self.size
        stackcor = np.zeros((self.patches, size_, size_))
        stackimg = np.zeros((self.patches, size_, size_))
        stackssim = np.zeros((self.patches, 1))
        trans = torchio.transforms.RandomGhosting(num_ghosts=int(np.random.randint(low=3,high=self.num_ghosts, size=1)[0]),#5,
                                         axes=np.random.randint(self.axes, size=1)[0],
                                         intensity=np.random.uniform(0.05, self.intensity, 1)[0],# 1.75,
                                         restore=np.random.uniform(0.01, self.restore, 1)[0])# 0.02)
        transMot = torchio.transforms.RandomMotion(degrees=np.random.uniform(0.01, self.degrees, 1)[0],# 10,
                                         translation=np.random.uniform(0.01, self.translation, 1)[0],# 10,
                                         num_transforms=int(np.random.randint(low=2,high=self.num_transforms, size=1)[0]),#5,
                                         image_interpolation='linear')

        for ii in range(0, self.patches):
            if self.modalityMotion == 0:
                img = select_slice(image_in_)
                img = cutNoise(img, self.level_noise)
                img = PadResize(img, size_)
                img = randomContrastAug(img)
                cor = self.cter(img)
                ssimtmp = structural_similarity(img, cor, data_range=1)
                stackcor[ii,:,:]= cor
                stackimg[ii,:,:]= img
                stackssim[ii,0] = ssimtmp
            elif self.modalityMotion == 1 :
                img = select_slice(image_in_)
                img = cutNoise(img, self.level_noise)
                img = PadResize(img, size_)
                img = randomContrastAug(img)
                ## corrupt with torchio:
                testtens = torch.unsqueeze(torch.unsqueeze(torch.tensor(img),0),0)
                d_ = transMot(testtens)
                d_ = trans(d_)
                cor = d_.detach().cpu().numpy().squeeze()
                ssimtmp = structural_similarity(img, cor, data_range=1)
                stackcor[ii,:,:]= cor
                stackimg[ii,:,:]= img
                stackssim[ii,0] = ssimtmp
            else: # if self.modalityMotion == 2:
                tempModMot = np.random.randint(2, size=1)[0]
                if tempModMot == 0:
                    # print(tempModMot)
                    img = select_slice(image_in_)
                    img = cutNoise(img, self.level_noise)
                    img = PadResize(img, size_)
                    img = randomContrastAug(img)
                    cor = self.cter(img)
                    ssimtmp = structural_similarity(img, cor, data_range=1)
                    stackcor[ii,:,:]= cor
                    stackimg[ii,:,:]= img
                    stackssim[ii,0] = ssimtmp
                else:
                    # print(tempModMot)
                    img = select_slice(image_in_)
                    img = cutNoise(img, self.level_noise)
                    img = PadResize(img, size_)
                    img = randomContrastAug(img)
                    ## corrupt with torchio:
                    testtens = torch.unsqueeze(torch.unsqueeze(torch.tensor(img),0),0)
                    d_ = transMot(testtens)
                    d_ = trans(d_)
                    cor = d_.detach().cpu().numpy().squeeze()
                    ssimtmp = structural_similarity(img, cor, data_range=1)
                    stackcor[ii,:,:]= cor
                    stackimg[ii,:,:]= img
                    stackssim[ii,0] = ssimtmp
            
        return stackcor, stackimg, stackssim

###########################################################################
###########################################################################
###########################################################################
def tensorboard_regression(writer, inputs, outputs, epoch, section='train'):
    writer.add_image('{}/output'.format(section),
                     vutils.make_grid(outputs[0, ...],
                                      normalize=True,
                                      scale_each=True),
                     epoch)
    if inputs is not None:
        writer.add_image('{}/input'.format(section),
                        vutils.make_grid(inputs[0, ...],
                                        normalize=True,
                                        scale_each=True),
                        epoch)

def getSSIM(gt, out, gt_flag, data_range=1):
    vals = []
    for i in range(gt.shape[0]):
        if not gt_flag[i]:
            continue
        for j in range(gt.shape[1]):
            vals.append(structural_similarity(gt[i,j,...], out[i,j,...], data_range=data_range))
    return median(vals)
