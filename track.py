import numpy as np
from scipy import interpolate, ndimage
import matplotlib.pyplot as plt

frames=np.load('dataFiles/4braccia.npy')
im=frames[t].copy()
#plt.figure()
#plt.imshow(im)
#plt.gray()
im=im.astype(float)
h,w=np.shape(im)

r1 = 50
r2 = 60

#---------------- Stretching Histogram -------------------

hstretch_low=0.05
hstretch_high=0.3


#stretch the histogram:
hcount,bine=np.histogram(im.flatten(),200)
binc=bine[:-1]+np.diff(bine)/2
hcount=hcount.astype(float)/(h*w)

#determine the low intensity treshhold:
mask=(np.cumsum(hcount)<hstretch_low)
ti=find(mask)[-1]
tresh_low=binc[ti]
#determine the high intensity treshhold:
mask=(np.cumsum(hcount)<hstretch_high)
ti=find(mask)[-1]
tresh_high=binc[ti]

im-=tresh_low
im[im<0]=0
im*=1/(tresh_high-tresh_low)
im[im>1]=1

#--------------- Create Ring ---------------------------

#plt.figure()
#plt.imshow(im)

#plt.figure()
#imshow(im>0.5)

m=im>0.5
x,y = ndimage.measurements.center_of_mass(~m)

xvec=np.arange(w)
yvec=np.arange(h)

X,Y=np.meshgrid(xvec-y,yvec-x)

rmat=sqrt(X**2+Y**2)

mask1=rmat>r1
mask2=rmat<r2
mask=logical_and(mask1,mask2)

im2=im=frames[t].copy().astype(float)

im2[~mask]=0
#plt.imshow(im2)

#--------------- Interpolation Ring ------------------------------

pval=im2[mask]
xc=X[mask]
yc=Y[mask]

theta=arctan2(yc,xc)
theta=theta*180/pi
r=rmat[mask]

tr = np.linspace(r1, r2, 20)
NP=round(2*pi*r2)
ttheta = np.linspace(-180,180,100)
RI,TI = np.meshgrid(tr,ttheta)

gridRot=interpolate.griddata((r,theta),pval,(RI,TI), fill_value=0)
I = sum(gridRot,axis=1)

FI=np.fft.rfft(I)

#rbf=Rbf(r,theta,function='gaussian',epsilon=0.01)

#ZI=rbf(RI,TI)

#plt.show()
