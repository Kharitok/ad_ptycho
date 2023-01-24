import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as nfft
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift, fftfreq

def ff(a):
    return(fftshift(nfft.fft2(ifftshift(a),norm = 'ortho')))

def get_circ_mask(size_x,size_y,cx,cy,r0):
    x = np.arange(0,size_x)
    y = np.arange(0,size_y)
    arr = np.zeros((y.size,x.size),dtype=bool)
    
    cx = cy
    cy = cx
    r = r0
    
    mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2
    arr[mask] = 1.0
    
    return(arr)
    
    
def get_ring_mask(size_x,size_y,cx,cy,r0,r1):
        x = np.arange(0,size_x)
        y = np.arange(0,size_y)
        arr = np.zeros((y.size,x.size),dtype=bool )

        cx = cy
        cy = cx
      

        mask = (r0**2<(x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2) * ((x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r1**2)
        arr[mask] = 1

        return(arr)
    
    
def one_bit_th(mask):
    return (0.5+2.4141*(1/np.sqrt(np.count_nonzero(mask)))) /(1.5+1.4142*(1/np.sqrt(np.count_nonzero(mask))))
    
def half_bit_th(mask):
    return (0.2071+1.9102*(1/np.sqrt(np.count_nonzero(mask))))/(1.207+0.9102*(1/np.sqrt(np.count_nonzero(mask))))

def two_sigma_th(mask):
    return 2/(np.sqrt(np.count_nonzero(mask)/2))

def calc_ring_corr(I1,I2):
    return (np.sum(I1*np.conj(I2))) /(np.sqrt( (np.sum((np.abs(I1))**2)) *((np.sum((np.abs(I2))**2)))))
    
def calc_FRC(I1,I2,bin_width=2):
    I1_f = ff(np.abs(I1))
    I2_f = ff(np.abs(I2))
    
    size = I1.shape[0]
    r_low = np.arange(0,size/2,bin_width)
    r_high = np.arange(0+bin_width,size/2+bin_width,bin_width)
    
    rs = []
    corrs = [] 
    ths = []
    
    c = I1_f*np.conj(I2_f)
    t1 = (np.abs(I1_f))**2
    t2 = (np.abs(I2_f))**2
    
    for r0,r1 in zip(r_low,r_high):
        mask = get_ring_mask(size,size,int(size/2),int(size/2),r0,r1)
        th = half_bit_th(mask)
        corr = np.real(  np.sum(c[mask])/(np.sqrt(np.sum(t1[mask])*np.sum(t2[mask]))))
        r = (r0+r1)/2
        
        rs.append(r)
        corrs.append(corr)
        ths.append(th)
#         print(r)
    
    return((np.array(rs),np.array(corrs),np.array(ths)))

def calc_FRC_from_one(I1,bin_width=2):
    a,b,c,d = split_to_4(I1)
    rs,corrs1,ths = calc_FRC(c,d,bin_width)
    rs,corrs2,ths = calc_FRC(a,b,bin_width)
    return((rs,(corrs1+corrs2)/2,ths))
    

def split_to_4(Img):
    Im1 =Img[:Img.shape[0]-1:2,:Img.shape[0]-1:2]
    Im2 =Img[1::2,1::2]
    Im3 = Img[1::2,:Img.shape[0]-1:2]
    Im4 =Img[:Img.shape[0]-1:2,1::2]
    return(Im1,Im2,Im3,Im4)

# def r_to_freq(r,dx,N):
from matplotlib_scalebar.scalebar import ScaleBar