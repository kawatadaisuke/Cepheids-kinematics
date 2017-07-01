
#
# verr_mc.py
#  estimating velocity error using MC sampling
#
#  30 June 2017 - written D. Kawata
#

import pyfits
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from scipy import stats
from galpy.util import bovy_coords

# read the data with velocity info.
infile='/Users/dkawata/work/obs/Cepheids/Genovali14/G14T34+TGAS+Gorynya.fits'
star_hdus=pyfits.open(infile)
star=star_hdus[1].data
star_hdus.close()
# number of data points
nstarv1=len(star['Mod'])
print 'number of stars from 1st file =',nstarv1
# name
name=star['Name_1']
# extract the necessary particle info
glonv=star['_Glon']
glatv=star['_Glat']
# rescaled Fe/H
fehv=star['__Fe_H_']
distv=np.power(10.0,(star['Mod']+5.0)/5.0)*0.001
modv=star['Mod']
moderrv=star['e_Mod']
# RA, DEC from Gaia data
rav=star['_RA']
decv=star['_DE']
pmrav=star['pmra']
pmdecv=star['pmdec']
errpmrav=star['pmra_error']
errpmdecv=star['pmdec_error']
pmradec_corrv=star['pmra_pmdec_corr']
hrvv=star['HRV']
errhrvv=star['e_HRV']
logp=star['logPer']
photnotes=star['Notes']

### read 2nd file
infile='/Users/dkawata/work/obs/Cepheids/Genovali14/G14T34+TGAS+Melnik15-Gorynya_wHRV.fits'
star_hdus=pyfits.open(infile)
star=star_hdus[1].data
star_hdus.close()
# read the 2nd data
# number of data points
nstarv2=len(star['Mod'])
print 'number of stars from 2nd file =',nstarv2
nstarv=nstarv1+nstarv2
print 'total number of stars =',nstarv
# name
name=np.hstack((name,star['Name']))
# extract the necessary particle info
glonv=np.hstack((glonv,star['_Glon']))
glatv=np.hstack((glatv,star['_Glat']))
# rescaled Fe/H
fehv=np.hstack((fehv,star['__Fe_H_']))
distv=np.hstack((distv,np.power(10.0,(star['Mod']+5.0)/5.0)*0.001))
modv=np.hstack((modv,star['Mod']))
moderrv=np.hstack((moderrv,star['e_Mod']))
# RA, DEC from Gaia data
rav=np.hstack((rav,star['_RA_1']))
decv=np.hstack((decv,star['_DE_1']))
pmrav=np.hstack((pmrav,star['pmra']))
pmdecv=np.hstack((pmdecv,star['pmdec']))
errpmrav=np.hstack((errpmrav,star['pmra_error']))
errpmdecv=np.hstack((errpmdecv,star['pmdec_error']))
pmradec_corrv=np.hstack((pmradec_corrv,star['pmra_pmdec_corr']))
hrvv=np.hstack((hrvv,star['HRV']))
errhrvv=np.hstack((errhrvv,star['e_HRV']))
logp=np.hstack((logp,star['logPer']))
photnotes=np.hstack((photnotes,star['Notes']))

# read the 3rd data
addDDO=True
# addDDO=False
if addDDO==True:
  infile='/Users/dkawata/work/obs/Cepheids/Genovali14/G14T34+TGAS+DDO16-Gorynya-Melnik15.fits'
# default HRV error
  HRVerr=2.0
  star_hdus=pyfits.open(infile)
  star=star_hdus[1].data
  star_hdus.close()
# number of data points
  nstarv3=len(star['Mod'])
  print 'number of stars from 3rd file =',nstarv3
  nstarv=nstarv1+nstarv2+nstarv3
  print 'total number of stars =',nstarv
# name
  name=np.hstack((name,star['Name']))
# extract the necessary particle info
  glonv=np.hstack((glonv,star['_Glon']))
  glatv=np.hstack((glatv,star['_Glat']))
# rescaled Fe/H
  fehv=np.hstack((fehv,star['__Fe_H_']))
  distv=np.hstack((distv,np.power(10.0,(star['Mod']+5.0)/5.0)*0.001))
  modv=np.hstack((modv,star['Mod']))
  moderrv=np.hstack((moderrv,star['e_Mod']))
# RA, DEC from Gaia data
  rav=np.hstack((rav,star['_RA']))
  decv=np.hstack((decv,star['_DE']))
  pmrav=np.hstack((pmrav,star['pmra']))
  pmdecv=np.hstack((pmdecv,star['pmdec']))
  errpmrav=np.hstack((errpmrav,star['pmra_error']))
  errpmdecv=np.hstack((errpmdecv,star['pmdec_error']))
  pmradec_corrv=np.hstack((pmradec_corrv,star['pmra_pmdec_corr']))
  hrvv=np.hstack((hrvv,star['RV_mean']))
  errhrvv=np.hstack((errhrvv,np.ones(nstarv3)*HRVerr))
  logp=np.hstack((logp,star['logPer']))
  photnotes=np.hstack((photnotes,star['Notes']))

# use galpy RA,DEC -> Glon,Glat
# Tlb=bovy_coords.radec_to_lb(rav,decv,degree=True,epoch=2000.0)
# degree to radian
glonradv=glonv*np.pi/180.0
glatradv=glatv*np.pi/180.0
pmvconst=4.74047
# convert proper motion from mu_alpha,delta to mu_l,b using bovy_coords
pmlonv=np.zeros(nstarv)
pmlatv=np.zeros(nstarv)
Tpmllbb=bovy_coords.pmrapmdec_to_pmllpmbb(pmrav,pmdecv,rav,decv,degree=True,epoch=2000.0)
# 
pmlonv=Tpmllbb[:,0]
pmlatv=Tpmllbb[:,1]
# pmlonv is pmlon x cons(b) 
vlonv=pmvconst*pmlonv*distv
vlatv=pmvconst*pmlatv*distv

### MC sampling
nmc=1001
# sample from proper-motion covariance matrix
pmradec_mc=np.empty((nstarv,2,nmc))
pmradec_mc[:,0,:]=np.atleast_2d(pmrav).T
pmradec_mc[:,1,:]=np.atleast_2d(pmdecv).T
for ii in range(nstarv):
  # constract covariance matrix
  tcov=np.zeros((2,2))
  tcov[0,0]=errpmrav[ii]**2.0/2.0  # /2 because of symmetrization below
  tcov[1,1]=errpmdecv[ii]**2.0/2.0
  tcov[0,1]=pmradec_corrv[ii]*errpmrav[ii]*errpmdecv[ii]
  # symmetrise
  tcov=(tcov+tcov.T)
  # Cholesky decomp.
  L=np.linalg.cholesky(tcov)
  pmradec_mc[ii]+=np.dot(L,np.random.normal(size=(2,nmc)))

pmra_samp=pmradec_mc[:,0,:]
plt.scatter(errpmrav,np.std(pmra_samp,axis=1))
plt.show()

# calculate errors
ratile=np.tile(rav,(nmc,1)).flatten()
dectile=np.tile(decv,(nmc,1)).flatten()
pmllbb_sam=bovy_coords.pmrapmdec_to_pmllpmbb(pmradec_mc[:,0,:].T.flatten() \
  ,pmradec_mc[:,1:].T.flatten(),ratile,dectile,degree=True,epoch=2000.0)
# reshape
pmllbb_sam=pmllbb_sam.reshape((nmc,nstarv,2))
# distance MC sampling 
mod_sam=np.random.normal(modv,moderrv,(nmc,nstarv))
# 
dist_sam=np.power(10.0,(mod_sam+5.0)/5.0)*0.001
dist_err=np.std(dist_sam,axis=0)
# check shape
print ' vlon, dist shape=',pmllbb_sam[:,:,0].shape,dist_sam.shape
# pmlonv is x cos(b) and vlat sample
vlon_sam=pmvconst*pmllbb_sam[:,:,0].flatten()*dist_sam.flatten()
vlon_sam=vlon_sam.reshape((nmc,nstarv))
vlat_sam=pmvconst*pmllbb_sam[:,:,1].flatten()*dist_sam.flatten()
vlat_sam=vlat_sam.reshape((nmc,nstarv))
# calculate errors
vlon_err=np.std(vlon_sam,axis=0)
vlat_err=np.std(vlat_sam,axis=0)

tbhdu = pyfits.BinTableHDU.from_columns([\
  pyfits.Column(name='Name',format='A20',array=name),\
  pyfits.Column(name='FeH',format='D',array=fehv),\
  pyfits.Column(name='Dist',format='D',array=distv), \
  pyfits.Column(name='e_Dist',format='D',array=dist_err), \
  pyfits.Column(name='Mod',format='D',array=modv), \
  pyfits.Column(name='e_Mod',format='D',array=moderrv), \
  pyfits.Column(name='Glon',format='D',array=glonv), \
  pyfits.Column(name='Glat',format='D',array=glatv), \
  pyfits.Column(name='RA',format='D',array=rav), \
  pyfits.Column(name='DEC',format='D',array=decv), \
  pyfits.Column(name='PMRA',format='D',array=pmrav), \
  pyfits.Column(name='e_PMRA',format='D',array=errpmrav), \
  pyfits.Column(name='PMDEC',format='D',array=pmdecv), \
  pyfits.Column(name='e_PMDEC',format='D',array=errpmdecv), \
  pyfits.Column(name='Vlon',format='D',array=vlonv), \
  pyfits.Column(name='e_Vlon',format='D',array=vlon_err), \
  pyfits.Column(name='Vlat',format='D',array=vlatv), \
  pyfits.Column(name='e_vlat',format='D',array=vlat_err), \
  pyfits.Column(name='HRV',format='D',array=hrvv), \
  pyfits.Column(name='e_HRV',format='D',array=errhrvv), \
  pyfits.Column(name='LogPer',format='D',array=logp)])
tbhdu.writeto('verr_mc.fits',clobber=True)

gs1=gridspec.GridSpec(2,1)
gs1.update(left=0.15,right=0.9,bottom=0.1,top=0.95,hspace=0,wspace=0)

# Vlon
plt.subplot(gs1[0])
# labes
plt.xlabel(r"Distance",fontsize=18,fontname="serif",style="normal")
plt.ylabel(r"$V_{lon}$",fontsize=18,fontname="serif",style="normal")
# scatter plot
plt.errorbar(distv,vlonv,xerr=dist_err,yerr=vlon_err,fmt='ok')

# Vlat
plt.subplot(gs1[1])
# labes
plt.xlabel(r"Distance",fontsize=18,fontname="serif",style="normal")
plt.ylabel(r"$V_{lat}$",fontsize=18,fontname="serif",style="normal")
# scatter plot
plt.errorbar(distv,vlatv,xerr=dist_err,yerr=vlon_err,fmt='ok')

plt.show()



