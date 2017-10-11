
#
# axsymdiskm-fit.py
#  fitting axisymmetric disk model to Cepheids kinematics data
#
#  10 Oct. 2017 - written D. Kawata
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
from emcee.utils import MPIPool
import corner
import sys
from scipy.misc import logsumexp
from mpi4py import MPI

comm=MPI.COMM_WORLD
rank=comm.Get_rank()

# define likelihood, constant Vc, sigR, Xsq
def lnlike(modelp,flags,fixvals,stardata):

# read flags
  hrhsig_fix,hrvsys_fit,dVcdR_fit,mcerrlike=flags

  if mcerrlike==True:
    n_s,nmc,glonrad_s,glatrad_s,hrv_sam,dist_sam,vlon_sam=stardata
  else:
    n_s,hrv_s,vlon_s,distxy_s,glonrad_s,errhrv_s,errvlon_s=stardata

# model parameters
  VcR0=modelp[0]
  Vphsun=modelp[1]
  Vrsun=modelp[2]
  sigrR0=modelp[3]
  Xsq=modelp[4]
  R0=modelp[5]
  ip=6
  if hrhsig_fix==True:
# fixed parameters
# radial density scale length is fixed
    hr=fixvals[0]
# radial velocity dispersion scale length is fixed
    hsig=fixvals[1]
  else:
    hr=fixvals[0]
    hsig=modelp[ip]
    ip+=1
  if hrvsys_fit==True:
    hrvsys=modelp[ip]
    ip+=1
  else:
    hrvsys=0.0
  if dVcdR_fit==True:
    dVcdR=modelp[ip]
  else:
    dVcdR=0.0

  if mcerrlike==True:
    glonrad_flsam=np.tile(glonrad_s,(nmc,1)).flatten()
    glatrad_flsam=np.tile(glatrad_s,(nmc,1)).flatten()
    dist_flsam=dist_sam.flatten()
    distxy_flsam=dist_flsam*np.cos(glatrad_flsam)
# velocity sample
# line-of-sight velocity
    hrvgal_flsam=hrv_sam.flatten()-hrvsys \
                 -Vrsun*np.cos(glonrad_flsam)+Vphsun*np.sin(glonrad_flsam)
# longitude velocity   
    vlongal_flsam=vlon_sam.flatten() \
                  +Vrsun*np.sin(glonrad_flsam)+Vphsun*np.cos(glonrad_flsam)
# calculate parameters at stellar position
    rgal_flsam=np.sqrt(R0**2+distxy_flsam**2-2.0*R0*distxy_flsam \
                              *np.cos(glonrad_flsam))
    phi_flsam=np.arccos((R0**2+rgal_flsam**2-distxy_flsam**2) \
                        /(2.0*R0*rgal_flsam))
    phi_flsam[glonrad_flsam>np.pi]=-phi_flsam[glonrad_flsam>np.pi]
    VcR_flsam=VcR0+(rgal_flsam-R0)*dVcdR
# asymmetric drift
    sigr_flsam=sigrR0*np.exp(-(rgal_flsam-R0)/hsig)
    Vasym_flsam=0.5*((sigr_flsam**2)/VcR_flsam)*(Xsq-1.0 \
      +rgal_flsam*(1.0/hr+2.0/hsig))
# expected mean hrvmean and dispersion
    hrvmean_flsam=(VcR_flsam-Vasym_flsam)*np.sin(phi_flsam+glonrad_flsam)
# expected mean vlonmean and dispersion
    vlonmean_flsam=(VcR_flsam-Vasym_flsam)*np.cos(phi_flsam+glonrad_flsam)
# sing(phi+l) and cos(phi+l)
    sphil=np.sin(phi_flsam+glonrad_flsam)
    cphil=np.cos(phi_flsam+glonrad_flsam)
# calculate deteminant 
    sigr_s2=sigr_flsam**2
    sigr_s4=sigr_flsam**4
    detv_flsam=(sigr_s4)*((sphil**2)*Xsq+cphil**2)*((cphil**2)*Xsq+sphil**2) \
        -(((cphil*sphil)*(Xsq-1.0))**2)*(sigr_s4)
# X^T V^-1 X
    xhrv=hrvgal_flsam-hrvmean_flsam
    xlon=vlongal_flsam-vlonmean_flsam
    xtvx_flsam=((xhrv**2)*(sigr_s2)*((cphil**2)*Xsq+sphil**2) \
        -xhrv*xlon*cphil*sphil*(Xsq-1.0)*(sigr_s2) \
        -xhrv*xlon*cphil*sphil*(sigr_s2)*(Xsq-1.0) \
        +(xlon**2)*(sigr_s2)*((sphil**2)*Xsq+cphil**2))/detv_flsam
# reshape the relevant variables
    detv_sam=detv_flsam.reshape((nmc,n_s))
    xtvx_sam=xtvx_flsam.reshape((nmc,n_s))

# log likelihood of each stars
    lnlkstar=logsumexp(-0.5*xtvx_sam \
                       ,axis=0,b=1.0/(2.0*np.pi*np.sqrt(detv_sam)))-np.log(nmc)

    lnlk=np.nansum(lnlkstar)

  else:
# stellar velocity in Galactic rest frame
# line-of-sight velocity
    hrvgal_s=hrv_s-hrvsys-Vrsun*np.cos(glonrad_s)+Vphsun*np.sin(glonrad_s)
# longitude velocity   
    vlongal_s=vlon_s+Vrsun*np.sin(glonrad_s)+Vphsun*np.cos(glonrad_s)
# calculate parameters at stellar position
    rgal_s=np.sqrt(R0**2+distxy_s**2-2.0*R0*distxy_s*np.cos(glonrad_s))
    phi_s=np.arccos((R0**2+rgal_s**2-distxy_s**2)/(2.0*R0*rgal_s))
    phi_s[glonrad_s>np.pi]=-phi_s[glonrad_s>np.pi]
    VcR_s=VcR0+(rgal_s-R0)*dVcdR
# asymmetric drift
    sigr_s=sigrR0*np.exp(-(rgal_s-R0)/hsig)
    Vasym_s=0.5*((sigr_s**2)/VcR_s)*(Xsq-1.0+rgal_s*(1.0/hr+2.0/hsig))
# sing(phi+l) and cos(phi+l)
    sphil=np.sin(phi_s+glonrad_s)
    cphil=np.cos(phi_s+glonrad_s)
# expected mean hrvmean 
    hrvmean_s=(VcR_s-Vasym_s)*sphil
# expected mean vlonmean 
    vlonmean_s=(VcR_s-Vasym_s)*cphil
# note: no error taken into account
# calculate deteminant 
    sigr_s2=sigr_s**2
    sigr_s4=sigr_s**4
    detv=(sigr_s4)*((sphil**2)*Xsq+cphil**2)*((cphil**2)*Xsq+sphil**2) \
        -(((cphil*sphil)*(Xsq-1.0))**2)*(sigr_s4)
# X^T V^-1 X
    xhrv=hrvgal_s-hrvmean_s
    xlon=vlongal_s-vlonmean_s
    xtvx=((xhrv**2)*(sigr_s2)*((cphil**2)*Xsq+sphil**2) \
        -xhrv*xlon*cphil*sphil*(Xsq-1.0)*(sigr_s2) \
        -xhrv*xlon*cphil*sphil*(sigr_s2)*(Xsq-1.0) \
        +(xlon**2)*(sigr_s2)*((sphil**2)*Xsq+cphil**2))/detv
# likelihood
    lnlk=np.nansum(-0.5*xtvx-np.log(2.0*np.pi*np.sqrt(detv)))

# radial velocity only
#  lnlk=np.nansum(-0.5*((hrvgal_s-hrvmean_s)**2/hrvsig2_s) \
#    -0.5*np.log(2.0*np.pi*hrvsig2_s))
# longitudinal velcoity only
#  lnlk=np.nansum(-0.5*((vlongal_s-vlonmean_s)**2/vlonsig2_s) \
#    -0.5*np.log(2.0*np.pi*vlonsig2_s))

# Reid et al. (2014) "conservative formula"
#  rhrv2ij=(hrvgal_s-hrvmean_s)**2/hrvsig2_s
#  rvlon2ij=(vlongal_s-vlonmean_s)**2/vlonsig2_s
#  lnlk=np.nansum(np.log((1.0-np.exp(-0.5*rhrv2ij))/rhrv2ij) \
#                 +np.log((1.0-np.exp(-0.5*rvlon2ij))/rvlon2ij))

  return lnlk

# define prior
def lnprior(modelp,flags,fixvals):

# read flags
  hrhsig_fix,hrvsys_fit,dVcdR_fit,mcerrlike=flags
# model parameters
  VcR0=modelp[0]
  Vphsun=modelp[1]
  Vrsun=modelp[2]
  sigrR0=modelp[3]
  Xsq=modelp[4]
  R0=modelp[5]
  ip=6
  if hrhsig_fix==True:
# fixed parameters
# radial density scale length is fixed
    hr=fixvals[0]
# radial velocity dispersion scale length is fixed
    hsig=fixvals[1]
  else:
    hr=fixvals[0]
    hsig=modelp[ip]
    ip+=1
  if hrvsys_fit==True:
    hrvsys=modelp[ip]
    ip+=1
  else:
    hrvsys=0.0
  if dVcdR_fit==True:
    dVcdR=modelp[ip]
  else: 
    dVcdR=0.0

  if VcR0<150.0 or VcR0>350.0 or Vphsun<150.0 or Vphsun>350.0 \
    or np.abs(Vrsun)>100.0 or sigrR0<0.0 or sigrR0>100.0 \
    or hsig<0.0 or hsig>800.0 or Xsq<0.0 or Xsq>100.0 or R0<0.0 \
    or R0>30.0 or np.abs(hrvsys)>100.0 or np.abs(dVcdR)>100.0:
    return -np.inf

  lnp=0.0
# Prior for R0 from Bland-Hawthorn & Gerhard (2016) 8.2pm0.1
  R0prior=8.2
  R0prior_sig=0.1
# Prior for R0 from Jo Bovy's recommendation on 28 June 2017
#  R0prior=8.1
#  R0prior_sig=0.1
#  R0prior_sig=0.4
# Prior for R0 from de Gris & Bono (2016)
#  R0prior=8.3
#  R0prior_sig=0.45

  lnp=-0.5*(R0-R0prior)**2/(R0prior_sig**2)-np.log(np.sqrt(2.0*np.pi)*R0prior_sig)

# prior for angular speed
# mocktest target
#  omgsun_prior=240.0/8.1
#  omgsun_prior=220.0/8.0
#  omgsun_prior_sig=0.12
# Bland-Hawthorn & Gerhard (2016)
  omgsun_prior=30.24
  omgsun_prior_sig=0.12

  omgsun=Vphsun/R0
  lnp=lnp-0.5*(omgsun-omgsun_prior)**2/(omgsun_prior_sig**2)-np.log(np.sqrt(2.0*np.pi)*omgsun_prior_sig)

# Xsq prior
#  Xsqprior=0.7
#  Xsqprior_sig=0.1  
#  lnp=lnp-0.5*(Xsq-Xsqprior)**2/(Xsqprior_sig**2)-np.log(np.sqrt(2.0*np.pi)*Xsqprior_sig)

  return lnp

# define the final ln probability
def lnprob(modelp,flags,fixvals,stardata):

  lp=lnprior(modelp,flags,fixvals)
  if not np.isfinite(lp):
    return -np.inf
  return lp+lnlike(modelp,flags,fixvals,stardata)

##### main programme start here #####

# flags
# use simulation data
# simdata=True
simdata=False
# use simulation data selected from the observed targets
# simdata_targets=True
simdata_targets=False
# mock data test using the location of input data
mocktest=True
# mocktest=False
# add V and distance error to mock data. 
# mocktest_adderr=True
mocktest_adderr=False

# mc sampling of likelihood take into account the errors
mcerrlike=True
# mcerrlike=False
# number of MC sample for Vlon sample
# nmc=1000
nmc=100

# only effective mcerrlike==False, take into account Verror only,
# ignore distance error
withverr=True
# withverr=False
if mcerrlike==True:
  withverr=True

# hr and hsig fix or not?
hrhsig_fix=True
# hrhsig_fix=False

# allow HRV systematic error
# only if hrhsig_fix, allow to explore hrvsys
if hrhsig_fix==True:
#  hrvsys_fit=True
  hrvsys_fit=False
else:
  hrvsys_fit=False

# fit dVcdR or not
dVcdR_fit=True
# dVcdR_fit=False

# set all flags
flags=hrhsig_fix,hrvsys_fit,dVcdR_fit,mcerrlike

# print flags
if rank==0:
  print ' hrhsig_fix,hrvsys_fit,dVcdR_fit,mcerrlike=',flags
  print ' withverr=',withverr
  print ' simdata=',simdata
  print ' mocktest=',mocktest
  print ' mocktest_adderr=',mocktest_adderr
  print ' mcerrlike=',mcerrlike

# fixed parameter
hr=4.0
if hrhsig_fix==True:
# fix hsig and hr
#  hsig=200.0
  hsig=10.0
  fixvals=np.zeros(3)
  fixvals[0]=hr
  fixvals[1]=hsig
  if rank==0:
    print ' fixed valuse hr,hsig=',hr,hsig
else:
  fixvals=np.zeros(1)
  fixvals[0]=hr
  if rank==0:
    print ' fixed valuse hr=',hr

# constant for proper motion unit conversion
pmvconst=4.74047

if simdata==True:
  ifile='lbsels.dat'
  # read sim data output from psy
  rdata=np.loadtxt(ifile,comments='#')
  print 'read file ',ifile
  xsim=rdata[:,0]
  ysim=rdata[:,1]
  zsim=rdata[:,2]
  vxsim=rdata[:,3]
  vysim=rdata[:,4]
  vzsim=rdata[:,5]
  glonsim=rdata[:,6]
  glatsim=rdata[:,7]
  d3dsim=rdata[:,8]
  vlonsim=rdata[:,9]
  hrvsim=rdata[:,10]
#  agesim=rdata[:,11]
  # selection
  zmaxlim=0.1
  sindx=np.where(zsim<zmaxlim)
  # set other values
  distxys=np.sqrt(xsim[sindx]**2+ysim[sindx]**2)
  if rank==0:
    print ' N selected particles=',len(xsim[sindx])
  glonrads=glonsim[sindx]*np.pi/180.0
  glatrads=glatsim[sindx]*np.pi/180.0
  vlons=vlonsim[sindx]
  hrvs=hrvsim[sindx]
  errvlons=np.zeros_like(vlons)
  errhrvs=np.zeros_like(hrvs)
  # unused set zero
  ras=np.zeros_like(hrvs)
  decs=np.zeros_like(hrvs)
  pmras=np.zeros_like(hrvs)
  errpmras=np.zeros_like(hrvs)
  pmdecs=np.zeros_like(hrvs)
  errpmdecs=np.zeros_like(hrvs)
  pmradec_corrs=np.zeros_like(hrvs)
  mods=np.zeros_like(hrvs)
  errmods=np.zeros_like(hrvs)
  logps=np.zeros_like(hrvs)
  names=map(str,glonsim[sindx])
elif simdata_targets==True:
  ifile='lbsels_targets.dat'
  # read sim data output from psy
  rdata=np.loadtxt(ifile,comments='#')
  if rank==0:
    print 'read file ',ifile
#  print ' 1st line=',rdata[0,:]
  glondegs=rdata[:,0]
  glonrads=rdata[:,0]*np.pi/180.0
  if rank==0:
    print ' N selected particles=',len(glonrads)
  glatdegs=rdata[:,1]
  glatrads=rdata[:,1]*np.pi/180.0
  distxys=rdata[:,2]
  hrvs=rdata[:,3]
  vlons=rdata[:,4]
  errhrvs=rdata[:,5]
  errvlons=rdata[:,6]
  mods=rdata[:,7]
  errmods=rdata[:,8]
  errpmras=rdata[:,22]
  errpmdecs=rdata[:,23]
  pmradec_corrs=rdata[:,24]
  logps=rdata[:,25]
  vlats=rdata[:,26]
  vxps=rdata[:,27]
  vyps=rdata[:,28]
  vzps=rdata[:,29]
  d3dps=rdata[:,30]
# get RA, DEC coordinates  
  Tradec=bovy_coords.lb_to_radec(glondegs,glatdegs,degree=True,epoch=2000.0)
  ras=Tradec[:,0]
  decs=Tradec[:,1]
# km/s to mas/yr
  pmlons=(vlons/d3dps/pmvconst)*np.cos(glatrads)
  pmlats=(vlats/d3dps/pmvconst)
  Tpmradec=bovy_coords.pmllpmbb_to_pmrapmdec(pmlons,pmlats,glondegs,glatdegs,degree=True,epoch=2000.0)
  pmras=Tpmradec[:,0]
  pmdecs=Tpmradec[:,1]
# velocity checks
#  Tvxyz=bovy_coords.vrpmllpmbb_to_vxvyvz(hrvs,pmlons,pmlats,glondegs,glatdegs,d3dps,degree=True)
#  f=open('lbsels_targets_vcheck.asc','w')
#  i=0
#  print >>f,"# nstar= %10d" % len(glonrads)
#  for i in range(len(glonrads)):
#    print >>f,"%12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e" \
#     %(glondegs[i],glatdegs[i],d3dps[i],vxps[i],vyps[i],vzps[i] \
#       ,Tvxyz[i,0],Tvxyz[i,1],Tvxyz[i,2])
#  f.close()  
#
  names=map(str,rdata[:,0])
else:
  # read verr_mc.py output
  infile='verr_mc.fits'
  star_hdus=pyfits.open(infile)
  star=star_hdus[1].data
  star_hdus.close()
  # number of data points
  nstarv=len(star['Mod'])
  # name
  name=star['Name']
  # extract the necessary particle info
  glonv=star['Glon']
  glatv=star['Glat']
  # rescaled Fe/H
#  fehv=star['FeH']
  modv=star['Mod']
  errmodv=star['e_Mod']
  distv=np.power(10.0,(modv+5.0)/5.0)*0.001
  # RA, DEC from Gaia data
  rav=star['RA']
  decv=star['DEC']
  pmrav=star['PMRA']
  pmdecv=star['PMDEC']
  errpmrav=star['e_PMRA']
  errpmdecv=star['e_PMDEC']
  pmradec_corrv=star['PMRADEC_corr']
  vlonv=star['Vlon']
  errvlonv=star['e_Vlon']
#  vlatv=star['Vlat']
#  errvlatv=star['e_Vlat']
  hrvv=star['HRV']
  errhrvv=star['e_HRV']
  logp=star['logPer']
  # radian glon and glat
  glonradv=glonv*np.pi/180.0
  glatradv=glatv*np.pi/180.0

  # z position
  zpos=distv*np.sin(glatradv)
  distxyv=distv*np.cos(glatradv)

  # select only velocity error is small enough
  # Verrlim=5.0
  # Verrlim=10.0
  Verrlim=10000.0
  zmaxlim=0.2
  # zmaxlim=1000.0
  distmaxlim=10.0
  zwerr=np.power(10.0,(modv+errmodv+5.0)/5.0)*0.001*np.sin(glatradv)
  sindx=np.where((np.sqrt(errvlonv**2+errhrvv**2)<Verrlim) & \
                 (np.abs(zwerr)<zmaxlim) & \
                 (distv<distmaxlim))
#                (distv<distmaxlim) & \
#                (logp>0.8))

  hrvs=hrvv[sindx]
  vlons=vlonv[sindx]
  distxys=distxyv[sindx]
  glonrads=glonradv[sindx]
  glatrads=glatradv[sindx]
  errvlons=errvlonv[sindx]
  errhrvs=errhrvv[sindx]
  ras=rav[sindx]
  decs=decv[sindx]
  pmras=pmrav[sindx]
  errpmras=errpmrav[sindx]
  pmdecs=pmdecv[sindx]
  errpmdecs=errpmdecv[sindx]
  pmradec_corrs=pmradec_corrv[sindx]
  mods=modv[sindx]
  errmods=errmodv[sindx]
  logps=logp[sindx]
  names=name[sindx]

# project HRV to radial velocity at b=0
hrvs=hrvs*np.cos(glatrads)
errhrvs=errhrvs*np.cos(glatrads)

nstars=len(hrvs)
if rank==0:
  print ' number of selected stars=',nstars  

nadds=0
if mcerrlike==True and nadds>0:
  if rank==0:
    print 'Error mcerrlike cannot have additional particles. nadds=',nadds
  sys.exit()

if mocktest==True and nadds>0:
# add or replace
  mock_add=False
  if mock_add==True:
# add more stars
    dmin=0.0
    dmax=4.0
    hrvadds=np.zeros(nadds)
    vlonadds=np.zeros(nadds)
    distxyadds=np.random.uniform(dmin,dmax,nadds)
    glonadds=np.random.uniform(0.0,2.0*np.pi,nadds)
# add the particles in the disk plane
    glatadds=np.zeros(nadds)
    hrvs=np.hstack((hrvs,hrvadds))
    vlons=np.hstack((vlons,vlonadds))
    distxys=np.hstack((distxys,distxyadds))
    glonrads=np.hstack((glonrads,glonadds))
    glatrads=np.hstack((glatrads,glatadds))
    errvhrvs=np.hstack((errhrvs,np.zeros(nadds)))
    errvlons=np.hstack((errvlons,np.zeros(nadds)))
    pmras=np.hstack((pmras,np.zeros(nadds)))
    errpmras=np.hstack((errpmras,np.zeros(nadds)))
    pmdecs=np.hstack((pmdecs,np.zeros(nadds)))
    errpmdecs=np.hstack((errpmdecs,np.zeros(nadds)))
    pmradec_corrs=np.hstack((pmradec_corrs,np.ones(nadds)))
    nstars=nstars+nadds
    if rank==0:
      print ' number of stars after addition of stars =',nstars  
  else:
# replace the particles.
    dmin=0.0
    dmax=4.0
    hrvadds=np.zeros(nadds)
    vlonadds=np.zeros(nadds)
# ramdomly homogeneous distribution
    distxyadds=np.sqrt(np.random.uniform(0.0,1.0,nadds))*dmax
#    distxyadds=np.random.uniform(dmin,dmax,nadds)
    glonadds=np.random.uniform(0.0,2.0*np.pi,nadds)
# add the particles in the disk plane
    glatadds=np.zeros(nadds)
    hrvs=hrvadds
    vlons=vlonadds
    distxys=distxyadds
    glonrads=glonadds
    glatrads=glatadds
    errhrvs=np.zeros(nadds)
    errvlons=np.zeros(nadds)
    pmras=np.zeros(nadds)
    errpmras=np.zeros(nadds)
    pmdecs=np.zeros(nadds)
    errpmdecs=np.zeros(nadds)
    pmradec_corrs=np.ones(nadds)
    nstars=nadds
    if rank==0:
      print ' number of stars after replacing =',nstars  

# output selected stars
if nadds==0 and rank==0:
  f=open('axsymdiskm-fit_sels.asc','w')
  i=0
  print >>f,"# nstar= %10d" % (nstars)
  for i in range(nstars):
    print >>f,"%12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %20s" \
     %(glonrads[i],glatrads[i],distxys[i],hrvs[i],vlons[i] \
     ,errhrvs[i],errvlons[i],mods[i],errmods[i] \
     ,ras[i],decs[i],pmras[i],pmdecs[i],errpmras[i],errpmdecs[i] \
     ,pmradec_corrs[i],logps[i],names[i])
  f.close()

### model fitting
# set initial model parameters
# default model parameters
nparam=6
# initial values
modelpname=np.array(['$V_c(R_0)$','$V_{\phi,\odot}$' \
  ,'$V_{R,\odot}$','$\sigma_R(R_0)$','$X^2$','$R_0$'])
# Bland-Hawthorn & Gerhard (2016), Vsun, V, Vrad
modelp0=np.array([237.2, 248.8, -10.0, 13.0, 1.0, 8.20])
# mw39
# modelp0=np.array([210.0, 220.0, -10.0, 30.0, 0.7, 8.0])
# for mock
# modelp0=np.array([230.0, 240.0, -8.0, 13.0, 0.8, 8.10])
# mwm 
# modelp0=np.array([210.0, 220.0, -10.0, 13.0, 0.55, 8.0])
# # local Vc is 204.7 at 8.04 kpc
# wrong parameters
# modelp0=np.array([210.0, 210.0, -10.0, 20.0, 1.0, 8.1])
if hrhsig_fix==False:
# fit hsig
  nparam+=1
  modelp0=np.hstack((modelp0,4.0))
  modelpname=np.hstack((modelpname,'$h_{\sigma}$'))
if hrvsys_fit==True:
  nparam+=1
  modelp0=np.hstack((modelp0,2.0))
  modelpname=np.hstack((modelpname,'$V_{los,sys}$'))
if dVcdR_fit==True:
  nparam+=1
  modelp0=np.hstack((modelp0,-3.6))
# mw39h10-1j
#  modelp0=np.hstack((modelp0,2.0))
  modelpname=np.hstack((modelpname,'$dV_c(R_0)/dR$'))

if rank==0:
  print ' N parameter fit=',nparam
  print ' parameters name=',modelpname

modelp=np.copy(modelp0)

# assign initial values for test output
# these will be used for target parameters for mock data
# model parameters
if mocktest==True:
  modelp0[0]=236.0
  modelp0[1]=247.968
  modelp0[2]=-9.0
  modelp0[3]=13.0
  modelp0[4]=1.0
  modelp0[5]=8.20
  if rank==0:
    print ' Mock test with reassigned velocity with true modelp=',modelp0

VcR0=modelp0[0]
Vphsun=modelp0[1]
Vrsun=modelp0[2]
sigrR0=modelp0[3]
Xsq=modelp0[4]
R0=modelp0[5]

ip=6
if hrhsig_fix==True:
# fixed parameters
# radial density scale length is fixed
  hr=fixvals[0]
# radial velocity dispersion scale length is fixed
  hsig=fixvals[1]
else:
  hr=fixvals[0]
  hsig=modelp[ip]
  ip+=1
if hrvsys_fit==True:
  hrvsys=modelp[ip]
  ip+=1
else:
  hrvsys=0.0
if dVcdR_fit==True:
  dVcdR=modelp[ip]
else: 
  dVcdR=0.0

xpos=-R0+np.cos(glonrads)*distxys
ypos=np.sin(glonrads)*distxys
rgals=np.sqrt(xpos**2+ypos**2)

if mocktest==True:
# assign the velocity using the true position
# test using mock data
# reassign hrvs, vlons
# exponential profile
  sigrs=sigrR0*np.exp(-(rgals-R0)/hsig)
  sigphs=np.sqrt(Xsq)*sigrs
  vrads=np.random.normal(0.0,sigrs,nstars)
  VcRs=VcR0+(rgals-R0)*dVcdR
# asymmetric drift
  Vasyms=0.5*((sigrs**2)/VcRs)*(Xsq-1.0+rgals*(1.0/hr+2.0/hsig))
  vphs=np.random.normal(VcRs-Vasyms,sigphs,nstars)
# angle from x=0, y=+
  angs=np.zeros(nstars)
  angs[ypos>=0]=np.arccos(-xpos[ypos>=0]/rgals[ypos>=0])
  angs[ypos<0]=2.0*np.pi-np.arccos(-xpos[ypos<0]/rgals[ypos<0])
  vxs=vphs*np.sin(angs)-vrads*np.cos(angs)
  vys=vphs*np.cos(angs)+vrads*np.sin(angs)
# no displacement in velocity
  hrvs=(vxs+Vrsun)*np.cos(glonrads)+(vys-Vphsun)*np.sin(glonrads)
  vlons=-(vxs+Vrsun)*np.sin(glonrads)+(vys-Vphsun)*np.cos(glonrads)
# set no Verr
#   errvlons=np.zeros(nstars)
#   errhrvs=np.zeros(nstars)
  if rank==0:
    f=open('axsymdiskm-fit_mock_input.asc','w')
    i=0
    for i in range(nstars):
      print >>f,"%f %f %f %f %f %f %f %f %f %f %f %f" %(xpos[i],ypos[i] \
       ,glonrads[i],rgals[i],vrads[i],vphs[i],angs[i],vxs[i],vys[i] \
       ,hrvs[i],vlons[i],Vasyms[i])
    f.close()

# output hrv and vlon input data and expected values from the above parameters
# line-of-sight velocity
hrvgals=hrvs-hrvsys-Vrsun*np.cos(glonrads)+Vphsun*np.sin(glonrads)
# longitude velocity
vlongals=vlons+Vrsun*np.sin(glonrads)+Vphsun*np.cos(glonrads)
# calculate parameters at stellar position
rgals=np.sqrt(R0**2+distxys**2-2.0*R0*distxys*np.cos(glonrads))
# rotation velocity
# th is angle between dxy and rgals
costh=(R0**2-rgals**2-distxys**2)/(-2.0*rgals*distxys)
sinth=np.sqrt(1.0-costh**2)
# radial and rotation velocity
vradgals=np.zeros_like(glonrads)
vrotgals=np.zeros_like(glonrads)
sindx=np.where(glonrads<=np.pi)
vradgals[sindx]=costh[sindx]*hrvgals[sindx]+sinth[sindx]*vlongals[sindx]
vrotgals[sindx]=sinth[sindx]*hrvgals[sindx]-costh[sindx]*vlongals[sindx]
sindx=np.where(glonrads>np.pi)
vradgals[sindx]=costh[sindx]*hrvgals[sindx]-sinth[sindx]*vlongals[sindx]
vrotgals[sindx]=-sinth[sindx]*hrvgals[sindx]-costh[sindx]*vlongals[sindx]
phis=np.arccos((R0**2+rgals**2-distxys**2)/(2.0*R0*rgals))
phis[ypos<0]=-phis[ypos<0]
VcRs=VcR0+(rgals-R0)*dVcdR
# expected mean hrvmean and dispersion
sigrs=sigrR0*np.exp(-(rgals-R0)/hsig)
Vasyms=0.5*((sigrs**2)/VcRs)*(Xsq-1.0+rgals*(1.0/hr+2.0/hsig))
hrvmeans=(VcRs-Vasyms)*np.sin(phis+glonrads)
hrvsig2s=(sigrR0**2)*(1.0+(np.sin(phis+glonrads)**2)*(Xsq-1.0))
# expected mean vlonmean and dispersion
vlonmeans=(VcRs-Vasyms)*np.cos(phis+glonrads)
vlonsig2s=(sigrR0**2)*(1.0+(np.cos(phis+glonrads)**2)*(Xsq-1.0))

# output ascii data for test
if rank==0:
  f=open('axsymdiskm-fit_hrvvlonmean_test.asc','w')
  i=0
  for i in range(nstars):
    print >>f,"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f" %(xpos[i],ypos[i] \
     ,glonrads[i],rgals[i],hrvs[i],vlons[i],hrvgals[i],vlongals[i] \
     ,phis[i],Vasyms[i],hrvmeans[i],np.sqrt(hrvsig2s[i]) \
     ,vlonmeans[i],np.sqrt(vlonsig2s[i]),vradgals[i],vrotgals[i])
  f.close()

if withverr==False:
# set errhrvs and errvlons zero 
  errhrvs=np.zeros(nstars)
  errvlons=np.zeros(nstars)

if mocktest_adderr==True:
  # add distance modulus error
  mods=np.random.normal(mods,errmods,nstars)
  dists=np.power(10.0,(mods+5.0)/5.0)*0.001
  distxys0=np.copy(distxys)
  distxys=dists*np.cos(glatrads)
  # add HRV error
  hrvs0=np.copy(hrvs)
  hrvs+=np.random.normal(0.0,errhrvs,nstars)
  # proper motion error adding
  pmradecs=np.empty((nstars,2))
  pmradecs[:,0]=pmras
  pmradecs[:,1]=pmdecs
  for ii in range(nstars):
    tcov=np.zeros((2,2))
    tcov[0,0]=errpmras[ii]**2.0/2.0  # /2 because of symmetrization below
    tcov[1,1]=errpmdecs[ii]**2.0/2.0
    tcov[0,1]=pmradec_corrs[ii]*errpmras[ii]*errpmdecs[ii]
    # symmetrise
    tcov=(tcov+tcov.T)
    # Cholesky decomp.
    L=np.linalg.cholesky(tcov)
    pmradecs[ii]+=np.dot(L,np.random.normal(size=(2)))
  if rank==0:
    f=open('mocktest_erroradded.asc','w')
    for i in range(nstars):
      print >>f,"%f %f %f %f %f %f %f %f %f %f %f %f" \
        %(distxys0[i],distxys[i] \
         ,hrvs0[i],hrvs[i],pmras[i],pmdecs[i],pmradecs[i,0],pmradecs[i,1] \
         ,(distxys[i]-distxys0[i]) \
         ,(hrvs[i]-hrvs0[i]) \
         ,(pmras[i]-pmradecs[i,0]) \
         ,(pmdecs[i]-pmradecs[i,1]))
    f.close()
  pmras=pmradecs[:,0]
  pmdecs=pmradecs[:,1]
  pmllbbs=bovy_coords.pmrapmdec_to_pmllpmbb(pmras,pmdecs,ras,decs \
    ,degree=True,epoch=2000.0)
  # pmlonv is x cos(b) and vlat sample
  vlons=pmvconst*pmllbbs[:,0]*dists
  vlats=pmvconst*pmllbbs[:,1]*dists

# set input star data
if mcerrlike==True:
  # sampled data for vlon
  # sample from proper-motion covariance matrix
  pmradec_mc=np.empty((nstars,2,nmc))
  pmradec_mc[:,0,:]=np.atleast_2d(pmras).T
  pmradec_mc[:,1,:]=np.atleast_2d(pmdecs).T
  for ii in range(nstars):
    # constract covariance matrix
    tcov=np.zeros((2,2))
    tcov[0,0]=errpmras[ii]**2.0/2.0  # /2 because of symmetrization below
    tcov[1,1]=errpmdecs[ii]**2.0/2.0
    tcov[0,1]=pmradec_corrs[ii]*errpmras[ii]*errpmdecs[ii]
    # symmetrise
    tcov=(tcov+tcov.T)
    # Cholesky decomp.
    L=np.linalg.cholesky(tcov)
    pmradec_mc[ii]+=np.dot(L,np.random.normal(size=(2,nmc)))
    # if ii==0:
    #  f=open('mcpmerradded_ii0.asc','w')
    #  print ' ii=0 correrr=',pmradec_corrs[ii]
    #  for i in range(nmc):
    #    print >>f,"%f %f" \
    #     %(pmradec_mc[ii,0,i],pmradec_mc[ii,1,i])
    #  f.close()
  # calculate errors
  ratile=np.tile(ras,(nmc,1)).flatten()
  dectile=np.tile(decs,(nmc,1)).flatten()
  pmllbb_sam=bovy_coords.pmrapmdec_to_pmllpmbb(pmradec_mc[:,0,:].T.flatten() \
    ,pmradec_mc[:,1:].T.flatten(),ratile,dectile,degree=True,epoch=2000.0)
  # reshape
  pmllbb_sam=pmllbb_sam.reshape((nmc,nstars,2))

  # distance MC sampling 
  mod_sam=np.random.normal(mods,errmods,(nmc,nstars))
  # test for no error
  # mod_sam=np.tile(mods,(nmc,1))
  # 
  dist_sam=np.power(10.0,(mod_sam+5.0)/5.0)*0.001
  dist_err=np.std(dist_sam,axis=0)
  # pmlonv is x cos(b) and vlat sample
  vlon_sam=pmvconst*pmllbb_sam[:,:,0].flatten()*dist_sam.flatten()
  vlon_sam=vlon_sam.reshape((nmc,nstars))
  vlat_sam=pmvconst*pmllbb_sam[:,:,1].flatten()*dist_sam.flatten()
  vlat_sam=vlat_sam.reshape((nmc,nstars))
  # calculate errors
  vlon_err=np.std(vlon_sam,axis=0)
  vlat_err=np.std(vlat_sam,axis=0)
  # MC sampling of hrv
  hrv_sam=np.random.normal(hrvs,errhrvs,(nmc,nstars))
  # test for no error
  # hrv_sam=np.tile(hrvs,(nmc,1))
  hrv_err=np.std(hrv_sam,axis=0)

  # plot error
#  gs1=gridspec.GridSpec(2,1)
#  gs1.update(left=0.15,right=0.9,bottom=0.1,top=0.95,hspace=0,wspace=0)
  # Vlon
#  plt.subplot(gs1[0])
  # labes
#  plt.xlabel(r"Distance",fontsize=18,fontname="serif",style="normal")
#  plt.ylabel(r"$V_{lon}$",fontsize=18,fontname="serif",style="normal")
  # scatter plot
#  plt.errorbar(distxys,vlons,xerr=dist_err,yerr=vlon_err,fmt='ok')
  # HRV
#  plt.subplot(gs1[1])
  # labes
#  plt.xlabel(r"Distance",fontsize=18,fontname="serif",style="normal")
#  plt.ylabel(r"HRV",fontsize=18,fontname="serif",style="normal")
  # scatter plot
#  plt.errorbar(distxys,hrvs,xerr=dist_err,yerr=hrv_err,fmt='ok')
#  plt.show()

  stardata=nstars,nmc,glonrads,glatrads,hrv_sam,dist_sam,vlon_sam
else:
  stardata=nstars,hrvs,vlons,distxys,glonrads,errhrvs,errvlons

# initial likelihood
lnlikeini=lnprob(modelp,flags,fixvals,stardata)

if rank==0:
  print ' Initial parameters=',modelp
  print ' Initial ln likelihood=',lnlikeini

# define number of dimension for parameters
# ndim,nwalkers=nparam,100
ndim,nwalkers=nparam,50
# initialise walker's position
# pos=[modelp+1.0e-3*np.random.randn(ndim) for i in range(nwalkers)]
# pos=[modelp+0.2*np.fabs(modelp)*np.random.randn(ndim) for i in range(nwalkers)]
pos=[modelp+np.fabs(modelp)*(-0.05+0.1*np.random.rand(ndim)) for i in range(nwalkers)]

pool=MPIPool()
if not pool.is_master():
  pool.wait()
  sys.exit(0)

# set up the sampler
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(flags,fixvals \
                                                           ,stardata),pool=pool)

# MCMC run
sampler.run_mcmc(pos,1000)
# sampler.run_mcmc(pos,500)

pool.close()

# burn in
samples=sampler.chain[:,200:,:].reshape((-1,ndim))

# save sampler data to file
if rank==0:
  np.save('sampler-chains.npy',sampler.chain[:,:,:])

# mean and standard deviation
mpmean=np.zeros(ndim)
mpstd=np.zeros(ndim)

i=0
while i<ndim:
  mpmean[i]=np.mean(samples[:,i])
  mpstd[i]=np.std(samples[:,i])
  if rank==0:
    print 'modelp',i,' mean,std= $%6.2f\pm %6.2f$' %(mpmean[i],mpstd[i])
  i+=1

if rank==0:
  f=open('mcmc_mean_std.asc','w')
  i=0
  for i in range(ndim):
    print >>f," %12.5e %12.5e" %(mpmean[i],mpstd[i])
  f.close()

# best-model likelihood
lnlikebf=lnprob(mpmean,flags,fixvals,stardata)

if rank==0:
  print ' Best model (MCMC mean)=',lnlikebf

# corner plot
# VcR0,Vphsun,Vrsun,sigrR0,hsig,Xsq,R0=modelp
if mocktest==True or simdata==True:
  fig = corner.corner(samples, \
      labels=modelpname,truths=modelp0)
else:
  fig = corner.corner(samples, \
                      labels=modelpname,truths=mpmean)
plt.show()

fig.savefig("modelparam.jpg")

