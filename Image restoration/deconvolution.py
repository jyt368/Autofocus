import numpy as np
from scipy.signal import convolve
import tifffile as tiff
import torch


def thresholdAndNorm(img, threshold=0.1,percentage=0.02, power=1):
    '''Threshold the image and normalize it
    Inputs:
    img: 3D numpy array of the image
    threshold: threshold value for the image
    percentage: percentage of pixels to saturate
    Output:
    img: 3D numpy array of the thresholded and normalized image'''
    # Saturate top 0.02% of pixels
    img = np.clip(img, 0, np.percentile(img, (100-percentage)))
    # Subtract the bottom 0.02% of pixels
    img = img - np.percentile(img, percentage)
    # Normalize the image
    img[img < 0] = 0
    img = img / np.amax(img)
    img[img < threshold] = 0

    # Enhance the contrast of the image
    img = np.power(img,power) # Increase dynamic range of sample 
    img = img-np.amin(img)
    img = (img/np.amax(img))
    return img

def taper_edges(img, psf):
    '''
    Taper the edges of the image with the PSF to reduce edge effects
    Inputs:
    img: 3D numpy array of the image
    psf: 2D numpy array of the PSF
    Output:
    img: 3D numpy array of the image with the edges tapered
    '''
    # Make a 2D array the size of the first 2 dimensions of the image
    taper = np.zeros(img.shape[:2])

    # Make the edges of the taper 1
    taper[:,0:3] = 1
    taper[:,-3:-1] = 1
    taper[0:3,:] = 1
    taper[-3:-1,:] = 1

    # Pad the psf to the size of the image
    padX = (img.shape[0] - psf.shape[0])/2
    padY = (img.shape[1] - psf.shape[1])/2
    psf = np.pad(psf, ((int(np.floor(padX)), int(np.ceil(padX))), (int(np.floor(padY)), int(np.ceil(padY)))), 'constant')
   

    # Blur taper with the PSF
    mask = (np.fft.fft2(taper))
    psf = (np.fft.fft2(psf))
    mask =np.fft.fftshift(np.fft.ifft2(mask * psf))
    mask = (mask).real
    
    # Normalize the mask
    mask = mask - np.amin(mask)
    mask = mask / np.amax(mask)

    # Apply the taper to the image
    for p in range(img.shape[2]):
        img[:,:,p] = img[:,:,p]*(1-mask)

    return img

def checkGPU():
    '''Check if GPU is available and return the device to be used for training and inference'''
    # Check if GPU is available
    if torch.cuda.is_available():
        print('CUDA is available. Using GPU')
        device = torch.device('cuda')
    else:
        print('CUDA is not available. Using CPU')
        print('Computation speed may be slow')
        device = torch.device('cpu')
    return device 

def richard_lucy_deconvolution(image_path, psf_path, n_it, device):
    # Load the image and PSF
    image = tiff.imread(image_path)
    psf = tiff.imread(psf_path)

    # Normalize the image and PSF
    img = image.astype(np.float32) / np.max(image)
    psf = psf.astype(np.float32) / np.sum(psf)

    # If device is GPU, clear the GPU cache
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()

    # Convert the image and PSF to torch tensors
    im_max = np.amax(img)
    img = torch.tensor(img).to(device)
    img = img / im_max

    # Normalize the PSF
    psf = psf / np.amax(psf)
    psf = torch.tensor(psf).to(device)
    psf = torch.fft.fftn(psf)

    estimate = img
    update = img

    # Start iterations
    for _ in range(n_it):
        estimate = update
        conv = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftn(estimate) * psf))
        relative_blur = img / conv
        error_estimation = torch.fft.ifftn(torch.fft.fftn(relative_blur) * torch.conj(psf))
        update = estimate * torch.fft.ifftshift(error_estimation.real)

    return update * im_max


# Check if GPU is available
device = checkGPU()
image_path = '.../resized_image.tif'
psf_path = '.../PSF.tif'
iterations = 20

result = richard_lucy_deconvolution(image_path, psf_path, iterations, device)
result = result.cpu().numpy()
result= thresholdAndNorm(result,0.01,0.001)
tiff.imsave('.../decon_result.tif', result)
