
#
# verr_mc.py
#  estimating velocity error using MC sampling
#
# History
#  15 May 2018 - D. Kawata
#    combine DR2 and Genovali+Melnik data. 
#  22 November 2017 - written D. Kawata
#  use only Genovali+Melnik data
#  
#

import pyfits
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from scipy import stats
from galpy.util import bovy_coords

# flag
# GaiaData = 'DR1'
GaiaData = 'DR2'

if GaiaData == 'DR1':
    # read the data with velocity info.
    infile='/Users/dkawata/work/obs/Cepheids/Genovali14/G14T34+TGAS+Melnik15.fits'
    star_hdus=pyfits.open(infile)
    star=star_hdus[1].data
    star_hdus.close()
    # select stars with HRV info
    sindx=np.where(star['r_HRV']>0) 
    # number of data points
    nstarv=np.size(sindx)
    print 'number of stars from 1st file =',nstarv
    # name
    name=star['Name'][sindx]
    # extract the necessary particle info
    glonv=star['_Glon'][sindx]
    glatv=star['_Glat'][sindx]
    # rescaled Fe/H
    fehv=star['__Fe_H_'][sindx]
    modv=star['Mod'][sindx]
    distv=np.power(10.0,(modv+5.0)/5.0)*0.001
    moderrv=star['e_Mod'][sindx]
    # RA, DEC from Gaia data
    rav=star['_RA_1'][sindx]
    decv=star['_DE_1'][sindx]
    pmrav=star['pmra'][sindx]
    pmdecv=star['pmdec'][sindx]
    errpmrav=star['pmra_error'][sindx]
    errpmdecv=star['pmdec_error'][sindx]
    pmradec_corrv=star['pmra_pmdec_corr'][sindx]
    hrvv=star['HRV'][sindx]
    errhrvv=star['e_HRV'][sindx]
    logp=star['logPer'][sindx]
    photnotes=star['Notes'][sindx]
else:
    nfiles = 2
    infile0 = '/Users/dkawata/work/obs/Cepheids/Genovali14/G14xGDR2d1xM15.fits'
    infile1 = '/Users/dkawata/work/obs/Cepheids/Genovali14/IYCep-combinedxM15.fits'
    star0 = pyfits.open(infile0)
    star1 = pyfits.open(infile1)
    nrows0 = star0[1].data.shape[0]
    nrows1 = star1[1].data.shape[0]
    nrows = nrows0 + nrows1
    star_hdu = pyfits.BinTableHDU.from_columns(star0[1].columns, nrows=nrows)
    for colname in star0[1].columns.names:
        star_hdu.data[colname][nrows0:] = star1[1].data[colname]
    star = star_hdu.data
    star0.close()
    star1.close()

    # select stars with HRV info
    sindx=np.where(star['r_HRV']>0) 
    # number of data points
    nstarv=np.size(sindx)
    print 'number of selected stars file =',nstarv
    # name
    name=star['name'][sindx]
    # extract the necessary particle info
    glonv=star['l'][sindx]
    glatv=star['b'][sindx]
    # rescaled Fe/H
    fehv=star['col__fe_h_'][sindx]
    modv=star['mod'][sindx]
    distv=np.power(10.0,(modv+5.0)/5.0)*0.001
    moderrv=star['e_mod'][sindx]
    # RA, DEC from Gaia data
    rav=star['ra'][sindx]
    decv=star['dec'][sindx]
    pmrav=star['pmra'][sindx]
    pmdecv=star['pmdec'][sindx]
    errpmrav=star['pmra_error'][sindx]
    errpmdecv=star['pmdec_error'][sindx]
    pmradec_corrv=star['pmra_pmdec_corr'][sindx]
    hrvv=star['HRV'][sindx]
    errhrvv=star['e_HRV'][sindx]
    logp=star['logper'][sindx]

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
nmc=10001
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

if GaiaData == 'DR1':
    outfile = 'verr_mc.fits'
else:
    outfile = 'verr_mc_gdr2.fits'

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
  pyfits.Column(name='PMRADEC_corr',format='D',array=pmradec_corrv), \
  pyfits.Column(name='Vlon',format='D',array=vlonv), \
  pyfits.Column(name='e_Vlon',format='D',array=vlon_err), \
  pyfits.Column(name='Vlat',format='D',array=vlatv), \
  pyfits.Column(name='e_vlat',format='D',array=vlat_err), \
  pyfits.Column(name='HRV',format='D',array=hrvv), \
  pyfits.Column(name='e_HRV',format='D',array=errhrvv), \
  pyfits.Column(name='LogPer',format='D',array=logp)])
tbhdu.writeto(outfile,clobber=True)

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



