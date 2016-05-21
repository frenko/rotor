import numpy as np
from scipy import interpolate, ndimage
import matplotlib.pyplot as plt
import matplotlib.mlab as lab

def load_image_from_file(filename='4braccia.npy', t=0, gray=0):
	'''
	Load image from numpy file.
	
	Parameters
	----------	
	filename: file name to load image
			  default 4braccia.npy
	
	t: frame number to load
		default 0
	
	gray: take a gray scale of image
		default 0
	
	Returns
	-------
	image: array of image
	'''
	path = 'dataFiles/'
	frames = np.load(path+filename)
	image = frames[t].copy()
	image = image.astype(float)
	
	if(gray == 1):
		plt.gray()
		
	return image
	
def histoStrech(img, hstretch_low=0.05, hstretch_high=0.3):
    '''
    Strech an histogram of taken image, in other word add a contrast to a given image
    
    Parameters
	----------
	img: image to stretch
	
	hstretch_low: (float) take a value of threshold low
		default 0.05
	
	hstretch_hight: (float) take a value of threshold high
		default 0.3
	
	Returns
	-------
	image: image array
	mask: mask array
    '''
    imgStr = img.copy()
    h,w = np.shape(imgStr)
    
    hstretch_low=hstretch_low
    hstretch_high=hstretch_high
    
    #stretch the histogram:
    hcount,bine=np.histogram(img.flatten(),200)
    binc=bine[:-1]+np.diff(bine)/2
    hcount=hcount.astype(float)/(h*w)

    #determine the low intensity treshhold:
    mask=(np.cumsum(hcount)<hstretch_low)
    ti=lab.find(mask)[-1]
    tresh_low=binc[ti]
    
    #determine the high intensity treshhold:
    mask=(np.cumsum(hcount)<hstretch_high)
    ti=lab.find(mask)[-1]
    tresh_high=binc[ti]

    imgStr-=tresh_low
    imgStr[imgStr<0]=0
    imgStr*=1/(tresh_high-tresh_low)
    imgStr[imgStr>1]=1
    
    return imgStr, mask

def createRing(img, imgOld, mask, r1=50, r2=65):
    '''
    Create a ring centered to center of mass of image with r1 of inner radius and r2 of outer radius
    
    Parameters
	----------
    img: is image to add ring
    
    imgOld: image without histo stretching
    
    r1: inner radius
    
    r2: outer radius
    
    mask: boolean mask
    
    Returns
    -------
    imageRinged: array of image
    X,Y: coordinates of ring
    '''
    
    height,width = np.shape(img)
    
    imgStr_Thresh = img>0.5
    
    y,x = ndimage.measurements.center_of_mass(~imgStr_Thresh)

    xvec=np.arange(width)
    yvec=np.arange(height)

    X,Y=np.meshgrid(xvec-x,yvec-y)

    rmat=np.sqrt(X**2+Y**2)

    mask1=rmat>r1
    mask2=rmat<r2
    mask=np.logical_and(mask1,mask2)

    imgRinged = imgOld.copy()

    imgRinged[~mask]=0
    return imgRinged, X, Y
    
def interpolateRing(img, mask, X, Y, r1=50, r2=65):
    '''
    Make an interpolation of ring with griddata function
    
    Parameters
	----------
    img: image to interpolate
    
    mask: bit mask
    
    X,Y: coordinates of ring
    
    r1, r2: inner and outer radius
    
    Returns
    -------
    I: array of pixel intensity
    '''
    #img[~mask] = 0
    
    pval=img[mask] #Values of pixel intensity
    
    #coordinates of ringed pixel
    xc=X[mask] 
    yc=Y[mask]

    theta=np.arctan2(yc,xc)
    theta=theta*180/np.pi
    rmat=np.sqrt(X**2+Y**2)
    
    
    r=rmat[mask]

    tr = np.linspace(r1, r2, 20)
    NP=round(2*np.pi*r2)
    ttheta = np.linspace(-180,180,100)
    RI,TI = np.meshgrid(tr,ttheta)

    gridRot=interpolate.griddata((r, theta), pval, (RI, TI), fill_value=0)
    I=np.sum(gridRot,axis=1)
    
    return I

