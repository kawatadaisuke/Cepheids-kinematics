
# plot MCMC results
# read sampler.npy from axsymdiskm-fit.py

import math
import numpy as np
import matplotlib.pyplot as plt
# import emcee
import corner


mocktest=True
# mocktest=False

ndim=7

modelp0=np.zeros(ndim)

if mocktest==True:
  modelp0[0]=236.0
  modelp0[1]=247.968
  modelp0[2]=-9.0
  modelp0[3]=13.0
  modelp0[4]=1.0
  modelp0[5]=8.20
  modelp0[6]=-3.6
  print ' Mock test with reassigned velocity with true modelp=',modelp0

schains=np.load('sampler-chains.npy')
samples=schains[:,200:,:].reshape((-1,ndim))

# mean and standard deviation
mpmean=np.zeros(ndim)
mpstd=np.zeros(ndim)

i=0
while i<ndim:
  mpmean[i]=np.mean(samples[:,i])
  mpstd[i]=np.std(samples[:,i])
  print 'modelp',i,' mean,std= $%6.2f\pm %6.2f$' %(mpmean[i],mpstd[i])
  i+=1

modelpname=np.array(['$V_c(R_0)$','$V_{\phi,\odot}$' \
                     ,'$V_{R,\odot}$','$\sigma_R(R_0)$','$X^2$','$R_0$' \
                     ,'$dV_c(R_0)/dR$'])

if mocktest==True or simdata==True:
  fig = corner.corner(samples, \
                      labels=modelpname,truths=modelp0, \
                      label_kwargs={"fontsize":16})
else:
  fig = corner.corner(samples,labels=modelpname,label_kwargs={"fontsize":16})

plt.show()

# fig.savefig("MCMC.jpg")
