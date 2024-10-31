# This code can obtain the PSF image and the corrected image from the input image with the estimated aberration parameters.
# sometimes power shouldn't be that large, or useful information might be lost. Power==1 is the original image not enhanced.
#Line 295, why add sample? Io = Io + sample (noise), what is the purpose of this line?
# Line 278-280, I add it. It seems the original code can't process stack images to get multiple PSF.
import numpy as np
import argparse
import matplotlib.pyplot as plt
from skimage import io, transform

import math
import os
import torch
print(torch.__version__)  # This line prints the PyTorch version
print(torch.version.cuda)  # This line prints the CUDA version
print(torch.cuda.device_count())
import torch.nn as nn
import time
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from models import GetModel
from tqdm import tqdm
import numpy as np
import glob
import argparse
import tifffile as tiff

def GetParams():
    opt = argparse.Namespace()

    opt.model='yolo'#'model to use'
    opt.lr = 0.0001 # learning rate
    opt.norm = 'minmax' # if normalization should not be used
    opt.nepoch =300 # number of epochs to train for
    opt.saveinterval =1 # number of epochs between saves
    opt.modifyPretrainedModel = False
    opt.multigpu = False
    opt.undomulti = False
    opt.ntrain = 2670 # number of samples to train on
    opt.scheduler = '' # options for a scheduler, format: stepsize,gamma
    opt.log = False
    opt.noise ='' # options for noise added, format: poisson,gaussVar

    # data
    opt.dataset = 'psf' # dataset to train
    opt.imageSize = '256' # the low resolution image size
    opt.weights = 'Z:/Users/yj368/aberration estimation/outputs/prelim300_test_Yuetong.pth' # model to retrain from
    opt.basedir = '' # path to prepend to all others paths: root, output, weights
    opt.root ='' # dataset to train
    opt.server = '' # whether to use server root preset
    opt.local = '' # whether to use local root preset
    opt.out = '' # folder to output model training results
    opt.n_zernike = 4 # number of Zernike modes used for PSF generation

    # computation 
    opt.workers  = 1 # number of data loading workers
    opt.batchSize = 5 # input batch size
    opt.accumulations = 10 # batch size multiplier

    # restoration options
    opt.task ='psf' # restoration task 
    opt.scale = 0 # low to high resolution scaling factor
    opt.nch_in = 5 # channels in input 
    opt.nch_out = 1 # channels in output 

    # architecture options 
    opt.narch = 0 # architecture-dependent parameter
    opt.n_resblocks  = 5 # number of residual blocks 
    opt.n_resgroups  = 5 # number of residual groups 
    opt.reduction  = 16 
    opt.n_feats = 128 # number of feature maps

    # test options
    opt.ntest  = 20 # number of images to test per epoch or test run 
    opt.testinterval  = 1 # number of epochs between tests during training 
    opt.test = False
    opt.cpu = False # not supported for training
    opt.batchSize_test  = 1 # input batch size for test loader 
    opt.plotinterval  = 1 # number of test samples between plotting 

    opt.NoiseLevel = 100 + 60*(np.random.rand()-0.5)
    # Poisson noise extent
    # include OTF and GT in stack
    opt.Poisson = np.random.randint(2000,6000)
    opt.OTF_and_GT = True
    # NA
    opt.NA = 1.49
    # Emission wavelength
    opt.emission = np.random.randint(500,690)
    opt.im_size = 256 # Model data size
    
    return opt

def PSF_conv(Io,PSF):
# Zero pad PSF around centre to match image size
    xysize = len(Io)
    pad = int((xysize-len(PSF))/2)
    PSF = np.pad(PSF,pad,'constant',constant_values=0)


    OTF = np.fft.fft2(PSF)
    OTF = np.fft.fftshift(OTF)

    #convolve Io wtih PSF
    Io = np.fft.fft2(Io)
    Io = np.fft.fftshift(Io)
    Io = Io*OTF
    Io = np.fft.ifftshift(Io)
    Io = np.fft.ifft2(Io)
    Io = np.fft.ifftshift(Io)
    Io = abs(Io)

    return Io

def threshold_and_norm(arr,power):

    arr = arr-np.amin(arr)
    arr = (arr/np.amax(arr))
    hist, bins = np.histogram(arr,16,[0, 1])
    ind = np.where(hist==np.amax(hist))
    mini = bins[ind[0][0]]
    print(mini)
    maxi = bins[ind[0][0]+1]
    print(maxi)
    sub = (maxi+mini)/2
    arr = arr - sub
    arr[arr<0]=0
    arr = (arr/np.amax(arr))
    arr = np.power(arr,power) # Increase dynamic range of sample 
    arr = arr-np.amin(arr)
    arr = (arr/np.amax(arr))
    return arr

def calcPSF(xysize,pixelSize,NA,emission,rindexObj,astig_1=0,astig_2=0,coma_1=0,coma_2=0,defocus=0,depletion=False):
    
    """
    Generate the aberrated incoherent emission PSF using A.Stokseth model.
    Parameters
    
    OUTPUT VARIABLES:
    psf: 2D array of incoherent PSF normalised between 0 and 1
    """

    #Calculated the wavelength of light inside the objective lens and specimen
    lambdaObj = emission/rindexObj

    #Calculate the wave vectors in vaccuum, objective and specimens
    kObj = 2*np.pi/lambdaObj

    #pixel size in frequency space
    dkxy = 2*np.pi/(pixelSize*xysize)

    #Radius of pupil
    kMax = (2*np.pi*NA)/(emission*dkxy)

    klims = np.linspace(-xysize/2,xysize/2,xysize)
    kx, ky = np.meshgrid(klims,klims)
    k = np.hypot(kx,ky)
    pupil = np.copy(k)
    pupil[pupil<kMax]=1
    pupil[pupil>=kMax]=0

    #sin of objective semi-angle
    sinthetaObj = (k*(dkxy))/(kObj)
    sinthetaObj[sinthetaObj>1] = 1

    costhetaObj = np.finfo(float).eps+np.sqrt(1-(sinthetaObj**2))

    #apodize the emission pupil
    pupil = (pupil/np.sqrt(costhetaObj))

    # get rho phi coordinates
    rho = (kx**2+ky**2)
    rho = rho/np.amax(rho)
    phi = np.arctan2(ky,kx)

    #calculate the astigmnatism phase mask
    astig_1_mask = astig_1*np.sqrt(6)*rho*np.sin(2*phi)
    astig_2_mask = astig_2*np.sqrt(6)*rho*np.cos(2*phi)
    
    #calculate the coma phase mask
    coma_1_mask = coma_1*np.sqrt(8)*(3*rho**3-2*rho)*np.sin(phi)
    coma_2_mask = coma_2*np.sqrt(8)*(3*rho**3-2*rho)*np.cos(phi)
    # wrap to 2pi   
    abb_mask = np.mod(astig_1_mask+astig_2_mask+coma_1_mask+coma_2_mask,2*np.pi)

    #Add spiral phase if depletion beam
    if depletion:
        depletion_mask = 0
    else:
        depletion_mask = 0

    #calculate zernike for defocus
    defocus_mask = defocus*np.sqrt(3)*(2*rho-1)
    #warap to 2pi
    defocus_mask = np.mod(defocus_mask,2*np.pi)

    #sum the masks  and wrap to 2pi
    abb_mask = np.mod(depletion_mask+abb_mask+defocus_mask,2*np.pi)
    
    #calculate the aberrated pupil
    pupilA = pupil*np.exp(1j*abb_mask)

    #calculate the coherent PSF
    psf = np.fft.ifft2(pupilA)

    #calculate the incoherent PSF
    psf = np.fft.fftshift(abs(psf**2))  

    return psf

def PSF_conv(Io,PSF):
    OTF = np.fft.fft2(PSF)
    OTF = np.fft.fftshift(OTF)

    #convolve Io wtih PSF
    Io = np.fft.fft2(Io)
    Io = np.fft.fftshift(Io)
    Io = Io*OTF
    Io = np.fft.fftshift(Io)
    Io = np.fft.ifft2(Io)
    Io = np.fft.ifftshift(Io)
    Io = abs(Io)
    Io = Io-np.amin(Io)
    Io = (Io/np.amax(Io))

    return Io





# ------------ Main loop --------------
imageDir = 'Z:/Users/yj368/Autofocus_repeat/Autofocus_repeat/ER_Yutong/repeat8.1/stacks'
#imageDir = 'C:/Users/yj368/ML_autofocus/autofocus/about/original'
n_rep = 33
planes = 5
start_num = 0
depletion = False
# get list of images in directory
imageList = glob.glob(imageDir+ '/*.tif')
bead_sample = False


opt = GetParams()
# Load the yolo model
net = GetModel(opt)
torch.cuda.empty_cache()

print('loading checkpoint', opt.weights)
checkpoint = torch.load(opt.weights)

net.load_state_dict(checkpoint['state_dict'])
net.eval()

PSF_stack = np.zeros((n_rep,opt.im_size,opt.im_size))

for i in range(n_rep):
    opt = GetParams()
    astig_1 = 0
    astig_2 = 0
    coma_1 = 0
    coma_2 = 0
    
    # higher power, harder to see the original image, power==1 is the original image not enhanced
    power = 1
    if bead_sample:
        sample = np.zeros((opt.im_size,opt.im_size))
        Rarr = np.random.rand(opt.im_size,opt.im_size)
        sample[Rarr<0.0006] = 1
        Io = sample
    else:

        file = imageList[i]    
        print('loading image sucess')
        Io = tiff.imread(file) 
        print('image shape', Io.shape)
        if len(Io.shape) == 2:
            Io = np.dstack((Io, Io, Io))  # Stack to make it RGB
            print('image shape', Io.shape)
        if  Io.shape[2] > 1:
            Io = Io.mean(2)

         
        minDim = np.amin(Io.shape)

        sample = np.zeros((opt.im_size,opt.im_size))
        Rarr = np.random.rand(opt.im_size,opt.im_size)
        sample[Rarr<0.0003] = 1

        Io = np.rot90(Io,i)
        Io = Io[0:minDim,0:minDim]
        Io = threshold_and_norm(Io,power)
        Io = transform.resize(Io, (opt.im_size, opt.im_size), anti_aliasing=True)
        Io = Io
        Io[Io>1]=1

        tiff.imwrite('Z:/Users/yj368/aberration estimation/Yuetong_test/stack' +str(i)+'_Io.tif',Io)

    Ia = np.zeros((planes,opt.im_size,opt.im_size))
    NoiseFrac = 80*np.random.rand()
    # generate the PSF for each plane
    for p in range(planes):
        defocus = 20-10*p
        PSF_excitation = calcPSF(opt.im_size,65,opt.NA,590,1.5,astig_1,astig_2,coma_1,coma_2,defocus,depletion)
        
        Ia[p,:,:] = PSF_conv(Io,PSF_excitation)

        poissonNoise = np.random.poisson(Ia[p,:,:]*opt.Poisson).astype(float)

        intensity = 1

        aNoise = opt.NoiseLevel/100  # noise
        nST = np.random.normal(0, aNoise*np.std(Ia[p,:,:], ddof=1), (Ia.shape[1],Ia.shape[2]))
            # may be set to 0 to avoid noise addition


        
        Ia[p,:,:] = intensity*Ia[p,:,:] + nST*NoiseFrac + poissonNoise
    
        Ia[p,:,:] = intensity*Ia[p,:,:]/np.amax(Ia[p,:,:])
    
    Ia = Ia-np.amin(Ia)
    Ia = Ia/np.amax(Ia)
    #convert to torch tensor
    Ia = torch.from_numpy(Ia)
    #convert to float
    Ia = Ia.float()
    #add batch dimension
    Ia = Ia.unsqueeze(0)
    sr = (net(Ia.cuda())).cpu().detach().numpy()
    # bring Ia back to numpy and cpu
    Ia = Ia.squeeze().numpy()

    print(sr)
    print(astig_1,astig_2,coma_1,coma_2)

    #get extimate params from sr
    sr = sr.squeeze()
    # get the differnece between the inoput and outputs
    est_astig_1 = astig_1-sr[0]
    est_astig_2 = astig_2-sr[1]
    est_coma_1 = coma_1-sr[2]
    est_coma_2 = coma_2-sr[3]
    print(est_astig_1,est_astig_2,est_coma_1,est_coma_2)

    #get the uncorrected PSF
    PSF_uncorrected = calcPSF(opt.im_size,65,opt.NA,590,1.5,astig_1,astig_2,coma_1,coma_2,0,True)

    #generate a corrected PSF from est
    PSF_est = calcPSF(opt.im_size,65,opt.NA,590,1.5,est_astig_1,est_astig_2,est_coma_1,est_coma_2,0,True)

    #get a predicted image from the corrected PSF
    corrected = PSF_conv(Io,PSF_est)

    '''# Assuming Ia is your image tensor and PSF_excitation is your PSF
    Ia_fft = np.fft.fft2(Ia[2,:,:].astype(float))
    PSF_fft = np.fft.fft2(PSF_est.astype(float))
    # Perform deconvolution in the Fourier domain, Add a small number to the denominator to avoid division by zero
    deconvolved_fft = Ia_fft / (PSF_fft+ 1e-7)
    # Convert back to the spatial domain
    deconvolved = np.fft.ifft2(deconvolved_fft)'''

    #normlaise images as 16-bit
    corrected = corrected/np.amax(corrected)
    corrected = corrected*65535
    corrected = np.uint16(corrected)

    Io = Io/np.amax(Io)
    Io = Io*65535
    Io = np.uint16(Io)
    '''Io[Io >= 65535] = 0  # Add this line
    print(np.any(Ia >= 65535))'''

    Ia = Ia.squeeze()
    Ia = Ia/np.amax(Ia)
    Ia = Ia*65535
    print(Ia.shape)
    Ia = np.uint16(Ia[2,:,:])
    '''Ia[Ia >= 65535] = 0  # Add this line
    print(np.any(Ia >= 65535))'''

    #upsample the PSFs, crop them and add them to the stack
    PSF_est = transform.resize(PSF_est, (opt.im_size, opt.im_size), anti_aliasing=True)
    PSF_est = PSF_est/np.amax(PSF_est)
    PSF_est = PSF_est*65535
    PSF_est = np.uint16(PSF_est)


    PSF_uncorrected = transform.resize(PSF_uncorrected, (opt.im_size, opt.im_size), anti_aliasing=True)
    PSF_uncorrected = PSF_uncorrected/np.amax(PSF_uncorrected)
    PSF_uncorrected = PSF_uncorrected*65535
    PSF_uncorrected = np.uint16(PSF_uncorrected)

    '''deconvolved_abs = np.abs(deconvolved)
    deconvolved_abs = transform.resize(deconvolved_abs, (opt.im_size, opt.im_size), anti_aliasing=True)
    deconvolved_abs = deconvolved_abs / np.amax(deconvolved_abs)
    deconvolved_abs = deconvolved_abs * 65535
    deconvolved_abs = np.uint16(deconvolved_abs)'''


    #make a stack of the images and PSFs
    stack = np.zeros((5,opt.im_size,opt.im_size))
    stack[4,:,:] = Io
    stack[2,:,:] = Ia
    stack[1,:,:] = PSF_est
    stack[0,:,:] = PSF_uncorrected
    stack[3,:,:] = corrected
    # stack[5,:,:] = deconvolved_abs

    # save the PSF in one stack
    PSF_stack[i] = stack[0,:,:]
 
#save the stack
tiff.imwrite('Z:/Users/yj368/aberration estimation/Yuetong_test/stack' +str(i)+'.tif',PSF_stack)

    



