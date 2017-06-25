
#
# axsymdiskm-fit.py
#  fitting axisymmetric disk model to Cepheids kinematics data
#
#  23 June 2017 - written D. Kawata
#

import pyfits
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from scipy import stats
from galpy.util import bovy_coords
import emcee
import corner

# define likelihood, constant Vc, sigR, Xsq
def lnlike(modelp,n_s,hrv_s,vlon_s,distxy_s,glonrad_s):
  
# model parameter
  VcR0,Vphsun,Vrsun,sigrR0,hsig,Xsq,R0=modelp
# radial density scale length is fixed
  hr=3.0

# stellar velocity in Galactic rest frame
# line-of-sight velocity
  hrvgal_s=hrv_s-Vrsun*np.cos(glonrad_s)+Vphsun*np.sin(glonrad_s)
# longitude velocity
  vlongal_s=vlon_s+Vrsun*np.sin(glonrad_s)+Vphsun*np.cos(glonrad_s)
#  vlongal_s=np.zeros(n_s)
#  for i in range(n_s):
#    if glonrad_s[i]<np.pi:
#      vlongal_s[i]=vlon_s[i]+Vrsun*np.sin(glonrad_s[i])+Vphsun*np.cos(glonrad_s[i])
#    else:
#      vlongal_s[i]=vlon_s[i]-Vrsun*np.sin(glonrad_s[i])+Vphsun*np.cos(glonrad_s[i])

# calculate parameters at stellar position
  rgal_s=np.sqrt(R0**2+distxy_s**2-2.0*R0*distxy_s*np.cos(glonrad_s))
  phi_s=np.arccos((R0**2+rgal_s**2-distxy_s**2)/(2.0*R0*rgal_s))
  phi_s[glonrad_s>np.pi]=-phi_s[glonrad_s>np.pi]
# asymmetric drift
  Vasym_s=0.5*((sigrR0**2)/VcR0)*(Xsq-1.0+rgal_s*(1.0/hr+2.0/hsig))
# expected mean hrvmean and dispersion
  hrvmean_s=(VcR0-Vasym_s)*np.sin(phi_s+glonrad_s)
  hrvsig2_s=(sigrR0**2)*(1.0+(np.sin(phi_s+glonrad_s)**2)*(Xsq-1.0))
# expected mean vlonmean and dispersion
  vlonmean_s=(VcR0-Vasym_s)*np.cos(phi_s+glonrad_s)
  vlonsig2_s=(sigrR0**2)*(1.0+(np.cos(phi_s+glonrad_s)**2)*(Xsq-1.0))

  lnlk=np.nansum(-0.5*((hrvgal_s-hrvmean_s)**2/hrvsig2_s \
    +(vlongal_s-vlonmean_s)**2/vlonsig2_s) \
    -np.log(2.0*np.pi*np.sqrt(hrvsig2_s)*np.sqrt(vlonsig2_s)))

#  rhrv2ij=(hrvgal_s-hrvmean_s)**2/hrvsig2_s
#  rvlon2ij=(vlongal_s-vlonmean_s)**2/vlonsig2_s
#
#  lnlk=np.nansum(np.log((1.0-np.exp(-0.5*rhrv2ij))/rhrv2ij) \
#                 +np.log((1.0-np.exp(-0.5*rvlon2ij))/rvlon2ij))
  return lnlk

# define prior
def lnprior(modelp):

# model parameter
  VcR0,Vphsun,Vrsun,sigrR0,hsig,Xsq,R0=modelp
  
  if VcR0<200.0 or VcR0>270.0 or Vphsun<200.0 or Vphsun>270.0 \
    or np.abs(Vrsun)>50.0 or sigrR0<0.0 or sigrR0>40.0 \
    or hsig<1.0 or hsig>800.0 or Xsq<0.0 or Xsq>10.0 or R0<6.0 \
    or R0>10.0:
    return -np.inf

  return 0.0

# define the final ln probability
def lnprob(modelp,n_s,hrv_s,vlon_s,distxy_s,glonrad_s):

  lp=lnprior(modelp)
  if not np.isfinite(lp):
    return -np.inf
  return lp+lnlike(modelp,n_s,hrv_s,vlon_s,distxy_s,glonrad_s)


##### main programme start here #####

# flags
mocktest=True

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
# RA, DEC from Gaia data
rav=star['_RA']
decv=star['_DE']
pmrav=star['pmra']
pmdecv=star['pmdec']
errpmrav=star['pmra_error']
errpmdecv=star['pmdec_error']
hrvv=star['HRV']
errhrvv=star['e_HRV']
logp=star['logPer']

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
# RA, DEC from Gaia data
rav=np.hstack((rav,star['_RA_1']))
decv=np.hstack((decv,star['_DE_1']))
pmrav=np.hstack((pmrav,star['pmra']))
pmdecv=np.hstack((pmdecv,star['pmdec']))
errpmrav=np.hstack((errpmrav,star['pmra_error']))
errpmdecv=np.hstack((errpmdecv,star['pmdec_error']))
hrvv=np.hstack((hrvv,star['HRV']))
errhrvv=np.hstack((errhrvv,star['e_HRV']))
logp=np.hstack((logp,star['logPer']))

# read the 3rd data
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
# RA, DEC from Gaia data
rav=np.hstack((rav,star['_RA']))
decv=np.hstack((decv,star['_DE']))
pmrav=np.hstack((pmrav,star['pmra']))
pmdecv=np.hstack((pmdecv,star['pmdec']))
errpmrav=np.hstack((errpmrav,star['pmra_error']))
errpmdecv=np.hstack((errpmdecv,star['pmdec_error']))
hrvv=np.hstack((hrvv,star['RV_mean']))
errhrvv=np.hstack((errhrvv,np.ones(nstarv3)*HRVerr))
logp=np.hstack((logp,star['logPer']))

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
# vx,vy
distxyv=distv*np.cos(glatradv)
# pmlonv is pmlon x cons(b) 
vlonv=pmvconst*pmlonv*distv
vlatxyv=pmvconst*pmlatv*distv*np.sin(glatradv)

# select only velocity error is small enough
Verrlim=5.0
errpmrav=pmvconst*distv*errpmrav
errpmdecv=pmvconst*distv*errpmdecv
sindx=np.where(np.sqrt(errpmrav**2+errpmdecv**2+errhrvv**2)<Verrlim)
# 
hrvs=hrvv[sindx]
vlons=vlonv[sindx]
distxys=distxyv[sindx]
glonrads=glonradv[sindx]
nstars=len(hrvs)
print ' number of selected stars=',nstars

### model fitting
# set initial model parameters
# VcR0,Vphsun,Vrsun,sigrR0,hsig,Xsq,R0=modelp
nparam=7
modelp0=np.array([240.0, 11.1+240.0, -11.1, 5.0, 4.0, 2.0, 8.34])
#modelp0=np.array([240.0, 11.1+240.0, -11.1, 5.0, 1.0, 2.0, 8.34])
modelp=modelp0
hr=3.0

if mocktest==True:
# test using mock data
# target parameters for mock data
# VcR0,Vphsun,Vrsun,sigrR0,hsig,Xsq,R0=modelp
  VcR0=modelp0[0]
  Vphsun=modelp0[1]
  Vrsun=modelp0[2]
  sigrR0=modelp0[3]
  hsig=modelp0[4]
  Xsq=modelp0[5]
  R0=modelp0[6]
  xpos=-R0+np.cos(glonrads)*distxys
  ypos=np.sin(glonrads)*distxys
  rgals=np.sqrt(xpos**2+ypos**2)
# asymmetric drift
  Vasyms=0.5*((sigrR0**2)/VcR0)*(Xsq-1.0+rgals*(1.0/hr+2.0/hsig))

# reassign hrvs, voons
  rgals=np.sqrt(xpos**2+ypos**2)
  sigrs=sigrR0
  sigphs=np.sqrt(Xsq)*sigrR0
  vrads=np.random.normal(0.0,sigrs,nstars)
  vphs=np.random.normal(VcR0-Vasyms,sigphs,nstars)
# angle from x=0, y=+
  angs=np.zeros(nstars)
  angs[ypos>=0]=np.arccos(-xpos[ypos>=0]/rgals[ypos>=0])
  angs[ypos<0]=2.0*np.pi-np.arccos(-xpos[ypos<0]/rgals[ypos<0])
  vxs=vphs*np.sin(angs)-vrads*np.cos(angs)
  vys=vphs*np.cos(angs)+vrads*np.sin(angs)
# re-set heliocentric velocity
  hrvs=(vxs+Vrsun)*np.cos(glonrads)+(vys-Vphsun)*np.sin(glonrads)
  vlons=-(vxs+Vrsun)*np.sin(glonrads)+(vys-Vphsun)*np.cos(glonrads)
  f=open('mock_input.asc','w')
  i=0
  for i in range(nstars):
    print >>f,"%f %f %f %f %f %f %f %f %f %f %f" %(xpos[i],ypos[i] \
     ,glonrads[i],rgals[i],vrads[i],vphs[i],angs[i],vxs[i],vys[i] \
     ,hrvs[i],vlons[i])
  f.close()

# output hrv and vlon input data and expected values from the above parameters
# line-of-sight velocity
  hrvgals=hrvs-Vrsun*np.cos(glonrads)+Vphsun*np.sin(glonrads)
# longitude velocity
  vlongals=vlons+Vrsun*np.sin(glonrads)+Vphsun*np.cos(glonrads)
# for i in range(nstars):
#  if glonrads[i]<np.pi:
#    vlongals[i]=vlons[i]+Vrsun*np.sin(glonrads[i])+Vphsun*np.cos(glonrads[i])
#  else:
#    vlongals[i]=vlons[i]-Vrsun*np.sin(glonrads[i])+Vphsun*np.cos(glonrads[i])
# calculate parameters at stellar position
  rgals=np.sqrt(R0**2+distxys**2-2.0*R0*distxys*np.cos(glonrads))
  phis=np.arccos((R0**2+rgals**2-distxys**2)/(2.0*R0*rgals))
  phis[ypos<0]=-phis[ypos<0]
# expected mean hrvmean and dispersion
  hrvmeans=(VcR0-Vasyms)*np.sin(phis+glonrads)
  hrvsig2s=(sigrR0**2)*(1.0+(np.sin(phis+glonrads)**2)*(Xsq-1.0))
# expected mean vlonmean and dispersion
  vlonmeans=(VcR0-Vasyms)*np.cos(phis+glonrads)
  vlonsig2s=(sigrR0**2)*(1.0+(np.cos(phis+glonrads)**2)*(Xsq-1.0))

# output ascii data for test
  f=open('hrvvlonmean_test.asc','w')
  i=0
  for i in range(nstars):
    print >>f,"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f" %(xpos[i],ypos[i] \
     ,glonrads[i],rgals[i],hrvs[i],vlons[i],hrvgals[i],vlongals[i] \
     ,rgals[i],phis[i],Vasyms[i],hrvmeans[i],np.sqrt(hrvsig2s[i]) \
     ,vlonmeans[i],np.sqrt(vlonsig2s[i]))
  f.close()

# initial likelihood
lnlikeini=lnprob(modelp,nstars,hrvs,vlons,distxys,glonrads)

print ' Initial parameters=',modelp
print ' Initial ln likelihood=',lnlikeini

# define number of dimension for parameters
ndim,nwalkers=nparam,100
# initialise walker's position
pos=[modelp+1.0e-3*np.random.randn(ndim) for i in range(nwalkers)]

# set up the sampler
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(nstars, \
  hrvs,vlons,distxys,glonrads))

# MCMC run
sampler.run_mcmc(pos,500)

# burn in
samples=sampler.chain[:,100:,:].reshape((-1,ndim))
# samples=sampler.chain[:,10:,:].reshape((-1,ndim))

# mean and standard deviation
mpmean=np.zeros(ndim)
mpstd=np.zeros(ndim)

i=0
while i<ndim:
  mpmean[i]=np.mean(samples[:,i])
  mpstd[i]=np.std(samples[:,i])
  print 'modelp',i,' mean,std=',mpmean[i],mpstd[i]
  i+=1

# corner plot
# VcR0,Vphsun,Vrsun,sigrR0,hsig,Xsq,R0=modelp
fig = corner.corner(samples, \
  labels=["$V_c(R_0)$","$V_{\phi,\odot}$","$V_{R,\odot}$", "$\sigma_R(R_0)$", \
     "$h_{\sigma}$","$X^2$","$R_0$"],truths=modelp0)

plt.show()

fig.savefig("modelparam.jpg")


