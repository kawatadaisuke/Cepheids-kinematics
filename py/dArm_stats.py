
#
# darm_stats.py
#
#  16 November 2017 - written D. Kawata
#

import pyfits
import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from scipy import stats
from galpy.util import bovy_coords
from galpy.util import bovy_plot

# define position of arm
def funcdarm2(x,armp,xposp,yposp):
# Arm parameters
  angref,rref,tanpa=armp
# Galactic parameters
  rsp=np.exp(tanpa*(x-angref))*rref

  darm2=(rsp*np.cos(x)-xposp)**2+(rsp*np.sin(x)-yposp)**2

#  print 'darm2,rsp,ang,xposp,yposp=',np.sqrt(darm2),rsp,x,xposp,yposp

  return darm2

# distance as a function of x


# define computing the distance to the arm
def distarm(armp,galp,xpos,ypos):
# Arm parameters
  angref,rref,tanpa=armp
# Galactic parameters
  rsun=galp

# find minimum 
  darm=np.zeros(len(xpos))
  angmin=np.zeros(len(xpos))

  for i in range(len(xpos)):
    res=minimize_scalar(funcdarm2,args=(armp,xpos[i],ypos[i]),bounds=(0.0,1.5*np.pi),method='bounded')
    if res.success==False:
      print ' no minimize result at i,x,y=',i,xpos[i],ypos[i]
      print ' res=',res
    darm[i]=np.sqrt(res.fun)
    angmin[i]=res.x

  return darm,angmin

##### starting main programme

# Sun's position
# Reid et al. (2014)
rsun=8.34
# Sun's proper motion Schoenrich et al.
usun=11.1
vsun=12.24
zsun=7.25
# Reid et al. (2014)
vcirc=240.0
# dvcdr=-3.6
dvcdr=0.0
print 'dVc/dR=',dvcdr

# read the data with velocity with MC error
# read verr_mc.py output
infile='verr_mc.fits'
star_hdus=pyfits.open(infile)
star=star_hdus[1].data
star_hdus.close()

# select stars
# select only velocity error is small enough
Verrlim=1000.0
zmaxlim=0.5
distmaxlim=10.0
zwerr=np.power(10.0,(star['Mod']+star['e_Mod']+5.0)/5.0)*0.001 \
  *np.sin(star['Glat']*np.pi/180.0)
verr=np.sqrt(star['e_Vlon']**2+star['e_Vlat']**2+star['e_HRV']**2)
dist=np.power(10.0,(star['Mod']+5.0)/5.0)*0.001
sindx=np.where((verr<Verrlim) & \
               (np.abs(zwerr)<zmaxlim) & \
               (dist<distmaxlim))


# name
namev=star['Name'][sindx]
# number of data points
nstarv=len(namev)
print ' number of stars selected=',nstarv
# extract the necessary particle info
glonv=star['Glon'][sindx]
glatv=star['Glat'][sindx]
fehv=star['FeH'][sindx]
modv=star['Mod'][sindx]
errmodv=star['e_Mod'][sindx]
distv=np.power(10.0,(modv+5.0)/5.0)*0.001
# RA, DEC from Gaia data
rav=star['RA'][sindx]
decv=star['DEC'][sindx]
pmrav=star['PMRA'][sindx]
pmdecv=star['PMDEC'][sindx]
errpmrav=star['e_PMRA'][sindx]
errpmdecv=star['e_PMDEC'][sindx]
pmradec_corrv=star['PMRADEC_corr'][sindx]
vlonv=star['Vlon'][sindx]
errvlonv=star['e_Vlon'][sindx]
vlatv=star['Vlat'][sindx]
errvlatv=star['e_Vlat'][sindx]
hrvv=star['HRV'][sindx]
errhrvv=star['e_HRV'][sindx]
logp=star['logPer'][sindx]
# radian glon and glat
glonradv=glonv*np.pi/180.0
glatradv=glatv*np.pi/180.0

# degree to radian, and conversion constant from as/yr to km/s
glonradv=glonv*np.pi/180.0
glatradv=glatv*np.pi/180.0
pmvconst=4.74047
# x, y position
xposv=-rsun+np.cos(glonradv)*distv*np.cos(glatradv)
yposv=np.sin(glonradv)*distv*np.cos(glatradv)
zposv=distv*np.sin(glatradv)
# rgal with Reid et al. value
rgalv=np.sqrt(xposv**2+yposv**2)

# -> vx,vy,vz
pmlonv=vlonv/(pmvconst*distv)
pmlatv=vlatv/(pmvconst*distv)
Tvxvyvz=bovy_coords.vrpmllpmbb_to_vxvyvz(hrvv,pmlonv,pmlatv,glonv,glatv,distv,degree=True)
vxv=Tvxvyvz[:,0]+usun
vyv=Tvxvyvz[:,1]+vsun
vzv=Tvxvyvz[:,2]+zsun
# Vcirc at the radius of the stars, including dVc/dR
vcircrv=vcirc+dvcdr*(rgalv-rsun)
vyv=vyv+vcircrv
# original velocity
vxv0=vxv
vyv0=vyv
vzv0=vzv
# Galactic radius and velocities
vradv0=(vxv0*xposv+vyv0*yposv)/rgalv
vrotv0=(vxv0*yposv-vyv0*xposv)/rgalv
# then subtract circular velocity contribution
vxv=vxv-vcircrv*yposv/rgalv
vyv=vyv+vcircrv*xposv/rgalv
vradv=(vxv*xposv+vyv*yposv)/rgalv
vrotv=(vxv*yposv-vyv*xposv)/rgalv

### compute distance from the arm
# Perseus
angen=(180.0-88.0)*np.pi/180.0
angst=(180.0+21.0)*np.pi/180.0
angref=(180.0-14.2)*np.pi/180.0
rref=9.9
# pitchangle
tanpa=np.tan(9.4*np.pi/180.0)

print ' ang range=',angst,angen

# set parameters
angrange=angst,angen
armp=angref,rref,tanpa
galp=rsun

# compute distances from the arm
darmv=np.zeros_like(xposv)
angarmv=np.zeros_like(xposv)
darmv,angarmv=distarm(armp,galp,xposv,yposv)

f=open('darm.asc','w')
i=0
rspv=np.exp(tanpa*(angarmv-angref))*rref
xarmp=rspv*np.cos(angarmv)
yarmp=rspv*np.sin(angarmv)
darmsunv=np.sqrt((xarmp+rsun)**2+yarmp**2)
distxyv=distv*np.cos(glatradv)
for i in range(nstarv):
#  if angarm[i]>angst and angarm[i]<angen:
  if distxyv[i]>darmsunv[i] and xposv[i]<-rsun:
    darmv[i]=-darmv[i]
  print >>f,"%f %f %f %f %f %f %f" %( \
    xposv[i],yposv[i],darmv[i],rspv[i]*np.cos(angarmv[i]) \
   ,rspv[i]*np.sin(angarmv[i]),angarmv[i]*180.0/np.pi,darmsunv[i])
f.close()

nrows=3
ncols=3
#gs1=gridspec.GridSpec(1,3)
# gs1.update(left=0.1,right=0.975,bottom=0.55,top=0.95)
# around the arm
warm=1.5
sindxarm=np.where((darmv>-warm) & (darmv<warm))
uradwarm=-vradv[sindxarm]
vrotwarm=vrotv[sindxarm]
darmwarm=darmv[sindxarm]

print ' number of stars |darm|<',warm,'=',len(uradwarm)
print ' U,V corrcoef=',np.corrcoef(darmwarm,uradwarm)[0,1] \
  ,np.corrcoef(darmwarm,vrotwarm)[0,1]

# plot d vs. U
plt.subplot(nrows,ncols,1)
plt.scatter(darmwarm,uradwarm,s=30)
plt.xlabel(r"d (kpc)",fontsize=12,fontname="serif")
plt.ylabel(r"U (km/s)",fontsize=12,fontname="serif")
plt.axis([-warm,warm,-80.0,80.0],'scaled')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plot d vs. V
# plt.subplot(gs1[1])
plt.subplot(nrows,ncols,2)
plt.scatter(darmwarm,vrotwarm,s=30)
plt.xlabel(r"d (kpc)",fontsize=12,fontname="serif")
plt.ylabel(r"V (km/s)",fontsize=12,fontname="serif")
plt.axis([-warm,warm,-80.0,80.0],'scaled')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# leading and training
dleadmax=-0.2
dleadmin=-1.5
dtrailmax=1.5
dtrailmin=0.2
sindxlead=np.where((darmv>dleadmin) & (darmv<dleadmax))
sindxtrail=np.where((darmv>dtrailmin) & (darmv<dtrailmax))

print ' Leading d range min,max=',dleadmin,dleadmax
print ' Trailing d range min,max=',dleadmin,dleadmax

# U is positive toward centre
uradl=-vradv[sindxlead]
uradt=-vradv[sindxtrail]
print ' numper of stars leading=',len(uradl)
print ' numper of stars trailing=',len(uradt)
# V
vrotl=vrotv[sindxlead]
vrott=vrotv[sindxtrail]

print ' Leading U med,mean,sig=',np.median(uradl),np.mean(uradl),np.std(uradl)
print ' Trailing U med,mean,sig=',np.median(uradt),np.mean(uradt),np.std(uradt)
print ' Leading V med,mean,sig=',np.median(vrotl),np.mean(vrotl),np.std(vrotl)
print ' Trailing V med,mean,sig=',np.median(vrott),np.mean(vrott),np.std(vrott)

# vertex deviation
# covariance of leading part
# leading part
v2dl=np.vstack((uradl,vrotl))
vcovl=np.cov(v2dl)
# print ' leading cov=',vcovl
# print ' std U, V=',np.sqrt(vcovl[0,0]),np.sqrt(vcovl[1,1])
print ' Leading vertex deviation=',(180.0/np.pi)*0.5*np.arctan(2.0*vcovl[0,1] \
 /(vcovl[0,0]-vcovl[1,1]))
# trailing part
v2dt=np.vstack((uradt,vrott))
vcovt=np.cov(v2dt)
# print ' Trailing cov=',vcovt
print ' Trailing vertex deviation=',(180.0/np.pi)*0.5*np.arctan(2.0*vcovt[0,1] \
 /(vcovt[0,0]-vcovt[1,1]))

# bottom panel
# U hist
# gs2=gridspec.GridSpec(1,3)
# gs2.update(left=0.1,right=0.975,bottom=0.1,top=0.4)
# plt.subplot(gs2[0])
plt.subplot(nrows,ncols,4)
plt.hist(uradt,bins=20,range=(-60,80),normed=True,histtype='step',color='b' \
 ,linewidth=2.0)
plt.hist(uradl,bins=20,range=(-60,80),normed=True,histtype='step',color='r' \
 ,linewidth=2.0)
plt.xlabel(r"U (km/s)",fontsize=12,fontname="serif")
plt.ylabel(r"dN",fontsize=12,fontname="serif")
# V hist
# plt.subplot(gs2[1])
plt.subplot(nrows,ncols,5)
plt.hist(vrott,bins=20,range=(-60,80),normed=True,histtype='step',color='b' \
 ,linewidth=2.0)
plt.hist(vrotl,bins=20,range=(-60,80),normed=True,histtype='step',color='r' \
 ,linewidth=2.0)
plt.xlabel(r"U (km/s)",fontsize=12,fontname="serif")
plt.ylabel(r"dN",fontsize=12,fontname="serif")
# U-V map
# leading
# plt.subplot(gs1[2])
plt.subplot(nrows,ncols,3)
plt.scatter(uradl,vrotl,s=30)
plt.xlabel(r"U (km/s)",fontsize=12,fontname="serif")
plt.ylabel(r"V (km/s)",fontsize=12,fontname="serif")
plt.axis([-60.0,60.0,-60.0,60.0],'scaled')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# trailing
# plt.subplot(gs2[2])
plt.subplot(nrows,ncols,6)
plt.scatter(uradt,vrott,s=30)
plt.xlabel(r"U (km/s)",fontsize=12,fontname="serif")
plt.ylabel(r"V (km/s)",fontsize=12,fontname="serif")
plt.axis([-60.0,60.0,-60.0,60.0],'scaled')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plot
plt.tight_layout()
plt.show()

##### MC error sampling
print '##### MC error sampling, velocity and distance'
# velocity and distance error only
nmc=100
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

# distribution of velocity and distance. 
# -> pml pmb
ratile=np.tile(rav,(nmc,1)).flatten()
dectile=np.tile(decv,(nmc,1)).flatten()
pmllbb_sam=bovy_coords.pmrapmdec_to_pmllpmbb(pmradec_mc[:,0,:].T.flatten() \
  ,pmradec_mc[:,1:].T.flatten(),ratile,dectile,degree=True,epoch=2000.0)
# reshape
pmllbb_sam=pmllbb_sam.reshape((nmc,nstarv,2))
# distance MC sampling 
modv_sam=np.random.normal(modv,errmodv,(nmc,nstarv))
# 
distv_sam=np.power(10.0,(modv_sam+5.0)/5.0)*0.001
# radial velocity MC sampling
hrvv_sam=np.random.normal(hrvv,errhrvv,(nmc,nstarv))
# -> vx,vy,vz
vxvyvz_sam=bovy_coords.vrpmllpmbb_to_vxvyvz(hrvv_sam.flatten() \
 ,pmllbb_sam[:,:,0].flatten(),pmllbb_sam[:,:,1].flatten() \
 ,np.tile(glonv,(nmc,1)).flatten(),np.tile(glatv,(nmc,1)).flatten() \
 ,distv_sam.flatten(),degree=True)
# reshape
vxvyvz_sam=vxvyvz_sam.reshape((nmc,nstarv,3))
vxv_sam=vxvyvz_sam[:,:,0]+usun
vyv_sam=vxvyvz_sam[:,:,1]+vsun
vzv_sam=vxvyvz_sam[:,:,2]+zsun
# x, y position
glonradv_sam=np.tile(glonradv,(nmc,1))
glatradv_sam=np.tile(glatradv,(nmc,1))
xposv_sam=-rsun+np.cos(glonradv_sam)*distv_sam*np.cos(glatradv_sam)
yposv_sam=np.sin(glonradv_sam)*distv_sam*np.cos(glatradv_sam)
zposv_sam=distv_sam*np.sin(glatradv_sam)
# rgal with Reid et al. value
rgalv_sam=np.sqrt(xposv_sam**2+yposv_sam**2)
# Vcirc at the radius of the stars, including dVc/dR
vcircrv_sam=vcirc+dvcdr*(rgalv_sam-rsun)
vyv_sam=vyv_sam+vcircrv_sam
# original velocity
vxv0_sam=vxv_sam
vyv0_sam=vyv_sam
vzv0_sam=vzv_sam
# Galactic radius and velocities
vradv0_sam=(vxv0_sam*xposv_sam+vyv0_sam*yposv_sam)/rgalv_sam
vrotv0_sam=(vxv0_sam*yposv_sam-vyv0_sam*xposv_sam)/rgalv_sam
# then subtract circular velocity contribution
vxv_sam=vxv_sam-vcircrv_sam*yposv_sam/rgalv_sam
vyv_sam=vyv_sam+vcircrv_sam*xposv_sam/rgalv_sam
vradv_sam=(vxv_sam*xposv_sam+vyv_sam*yposv_sam)/rgalv_sam
vrotv_sam=(vxv_sam*yposv_sam-vyv_sam*xposv_sam)/rgalv_sam

f=open('mcsample.asc','w')
for j in range(nstarv):
  for i in range(nmc):
    print >>f,"%f %f %f %f %f %f %f %f" %( \
     xposv_sam[i,j],yposv_sam[i,j],xposv[j],yposv[j] \
    ,vradv_sam[i,j],vrotv_sam[i,j],vradv[j],vrotv[j])
f.close()

# compute distances from the arm
darmv_sam=np.zeros_like(xposv_sam.flatten())
angarmv_sam=np.zeros_like(xposv_sam.flatten())
darmv_sam,angarmv_sam=distarm(armp,galp,xposv_sam.flatten(),yposv_sam.flatten())
# reshape
darmv_sam=darmv_sam.reshape((nmc,nstarv))
angarmv_sam=angarmv_sam.reshape((nmc,nstarv))

# check the position relative to the arm
rspv_sam=np.exp(tanpa*(angarmv_sam-angref))*rref
xarmp_sam=rspv_sam*np.cos(angarmv_sam)
yarmp_sam=rspv_sam*np.sin(angarmv_sam)
darmsunv_sam=np.sqrt((xarmp_sam+rsun)**2+yarmp_sam**2)
distxyv_sam=distv_sam*np.cos(glatradv_sam)
for j in range(nstarv):
  for i in range(nmc):
    if distxyv_sam[i,j]>darmsunv_sam[i,j] and xposv_sam[i,j]<-rsun:
      darmv_sam[i,j]=-darmv_sam[i,j]

# mean value
darmv_mean=np.mean(darmv_sam,axis=0).reshape(nstarv)
angarmv_mean=np.mean(angarmv_sam,axis=0).reshape(nstarv)
vradv_mean=np.mean(vradv_sam,axis=0).reshape(nstarv)
vrotv_mean=np.mean(vrotv_sam,axis=0).reshape(nstarv)

# sampling the stars around the arm
nrows=3
ncols=3
warm=1.5
sindxarm=np.where((darmv_mean>-warm) & (darmv_mean<warm))
nswarm=np.size(sindxarm)
print ' sindxarm=',nswarm
uradwarm_sam=-vradv_sam[:,sindxarm].reshape(nmc,nswarm)
vrotwarm_sam=vrotv_sam[:,sindxarm].reshape(nmc,nswarm)
darmwarm_sam=darmv_sam[:,sindxarm].reshape(nmc,nswarm)

# mean value

print ' number of stars selected from d_mean<',warm,'=',nswarm
uradwarm_mean=np.mean(uradwarm_sam,axis=0).reshape(nswarm)
vrotwarm_mean=np.mean(vrotwarm_sam,axis=0).reshape(nswarm)
darmwarm_mean=np.mean(darmwarm_sam,axis=0).reshape(nswarm)
uradwarm_std=np.std(uradwarm_sam,axis=0).reshape(nswarm)
vrotwarm_std=np.std(vrotwarm_sam,axis=0).reshape(nswarm)
darmwarm_std=np.std(darmwarm_sam,axis=0).reshape(nswarm)

# computing correlation coefficients
ucorrcoef=np.zeros(nmc)
vcorrcoef=np.zeros(nmc)
for i in range(nmc):
  ucorrcoef[i]=np.corrcoef(darmwarm_sam[i,:],uradwarm_sam[i,:])[0,1]
  vcorrcoef[i]=np.corrcoef(darmwarm_sam[i,:],vrotwarm_sam[i,:])[0,1]

print ' U corrcoef med,mean,sig=',np.median(ucorrcoef) \
  ,np.mean(ucorrcoef),np.std(ucorrcoef)
print ' V corrcoef med,mean,sig=',np.median(vcorrcoef) \
 ,np.mean(vcorrcoef),np.std(vcorrcoef)

# plot d vs. U
plt.subplot(nrows,ncols,1)
# plt.scatter(darmwarm_mean,uradwarm_mean,s=30)
plt.errorbar(darmwarm_mean,uradwarm_mean,xerr=darmwarm_std,yerr=uradwarm_std,fmt='.')
plt.xlabel(r"d (kpc)",fontsize=12,fontname="serif")
plt.ylabel(r"U (km/s)",fontsize=12,fontname="serif")
plt.axis([-warm,warm,-80.0,80.0],'scaled')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plot d vs. V
# plt.subplot(gs1[1])
plt.subplot(nrows,ncols,2)
# plt.scatter(darmwarm_mean,vrotwarm_mean,s=30)
plt.errorbar(darmwarm_mean,vrotwarm_mean,xerr=darmwarm_std,yerr=vrotwarm_std,fmt='.',)
plt.xlabel(r"d (kpc)",fontsize=12,fontname="serif")
plt.ylabel(r"V (km/s)",fontsize=12,fontname="serif")
plt.axis([-warm,warm,-80.0,80.0],'scaled')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# leading and training
sindxlead=np.where((darmv_mean>dleadmin) & (darmv_mean<dleadmax))
sindxtrail=np.where((darmv_mean>dtrailmin) & (darmv_mean<dtrailmax))

print ' Leading d range min,max=',dleadmin,dleadmax
print ' Trailing d range min,max=',dleadmin,dleadmax
nlead=np.size(sindxlead)
ntrail=np.size(sindxtrail)
print ' numper of stars leading=',nlead
print ' numper of stars trailing=',ntrail

# U is positive toward centre
uradl_sam=-vradv_sam[:,sindxlead].reshape(nmc,nlead)
uradt_sam=-vradv_sam[:,sindxtrail].reshape(nmc,ntrail)
# V
vrotl_sam=vrotv_sam[:,sindxlead].reshape(nmc,nlead)
vrott_sam=vrotv_sam[:,sindxtrail].reshape(nmc,ntrail)

# computing mean and median U, V
uradl_mean_sam=np.mean(uradl_sam,axis=1)
uradt_mean_sam=np.mean(uradt_sam,axis=1)
vrotl_mean_sam=np.mean(vrotl_sam,axis=1)
vrott_mean_sam=np.mean(vrott_sam,axis=1)
uradl_median_sam=np.median(uradl_sam,axis=1)
uradt_median_sam=np.median(uradt_sam,axis=1)
vrotl_median_sam=np.median(vrotl_sam,axis=1)
vrott_median_sam=np.median(vrott_sam,axis=1)

# taking statistical mean and dispersion
print '### Statistical mean and dispersion for Mean'
print ' Leading U mean,sig=',np.mean(uradl_mean_sam),np.std(uradl_mean_sam)
print ' Trailing U mean,mean,sig=',np.mean(uradt_mean_sam),np.std(uradt_mean_sam)
print ' Leading V mean,sig=',np.mean(vrotl_mean_sam),np.std(vrotl_mean_sam)
print ' Trailing V mean,sig=',np.mean(vrott_mean_sam),np.std(vrott_mean_sam)
print '### Statistical mean and dispersion for Median'
print ' Leading U mean,sig=',np.mean(uradl_median_sam),np.std(uradl_median_sam)
print ' Trailing U mean,mean,sig=',np.mean(uradt_median_sam),np.std(uradt_median_sam)
print ' Leading V mean,sig=',np.mean(vrotl_median_sam),np.std(vrotl_median_sam)
print ' Trailing V mean,sig=',np.mean(vrott_median_sam),np.std(vrott_median_sam)

# vertex deviation
# covariance of leading part
# leading part
lvl_sam=np.zeros(nmc)
for i in range(nmc):
  v2dl=np.vstack((uradl_sam[i,:],vrotl_sam[i,:]))
  vcovl=np.cov(v2dl)
  # vertex deviation
  lvl_sam[i]=(180.0/np.pi)*0.5*np.arctan(2.0*vcovl[0,1] \
    /(vcovl[0,0]-vcovl[1,1]))

print '### Vertex deviation' 
print ' Leading vertex deviation mean,sig=',np.mean(lvl_sam),np.std(lvl_sam)

# trailing part
lvt_sam=np.zeros(nmc)
for i in range(nmc):
  v2dt=np.vstack((uradt_sam[i,:],vrott_sam[i,:]))
  vcovt=np.cov(v2dt)
  lvt_sam[i]=(180.0/np.pi)*0.5*np.arctan(2.0*vcovt[0,1] \
   /(vcovt[0,0]-vcovt[1,1]))

print ' Trailing vertex deviation mean,sig=',np.mean(lvt_sam),np.std(lvt_sam)

# bottom panel
# U hist
# gs2=gridspec.GridSpec(1,3)
# gs2.update(left=0.1,right=0.975,bottom=0.1,top=0.4)
# plt.subplot(gs2[0])
plt.subplot(nrows,ncols,4)
plt.hist(uradt_sam.flatten(),bins=40,range=(-60,80) \
  ,normed=True,histtype='step',color='b',linewidth=2.0)
plt.hist(uradl_sam.flatten(),bins=40,range=(-60,80) \
  ,normed=True,histtype='step',color='r',linewidth=2.0)
plt.xlabel(r"U (km/s)",fontsize=12,fontname="serif")
plt.ylabel(r"dN",fontsize=12,fontname="serif")
# V hist
# plt.subplot(gs2[1])
plt.subplot(nrows,ncols,5)
plt.hist(vrott_sam.flatten(),bins=40,range=(-60,80) \
  ,normed=True,histtype='step',color='b',linewidth=2.0)
plt.hist(vrotl_sam.flatten(),bins=20,range=(-60,80) \
  ,normed=True,histtype='step',color='r',linewidth=2.0)
plt.xlabel(r"U (km/s)",fontsize=12,fontname="serif")
plt.ylabel(r"dN",fontsize=12,fontname="serif")
# U-V map
# leading
# plt.subplot(gs1[2])
plt.subplot(nrows,ncols,3)
# plt.scatter(uradl,vrotl,s=30)

plt.errorbar(np.mean(uradl_sam,axis=0).reshape(nlead) \
  ,np.mean(vrotl_sam,axis=0).reshape(nlead) \
  ,xerr=np.std(uradl_sam,axis=0).reshape(nlead) \
  ,yerr=np.std(vrotl_sam,axis=0).reshape(nlead),fmt='.')
plt.xlabel(r"U (km/s)",fontsize=12,fontname="serif")
plt.ylabel(r"V (km/s)",fontsize=12,fontname="serif")
plt.axis([-60.0,60.0,-60.0,60.0],'scaled')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# trailing
# plt.subplot(gs2[2])
plt.subplot(nrows,ncols,6)
# plt.scatter(uradt,vrott,s=30)
plt.errorbar(np.mean(uradt_sam,axis=0).reshape(ntrail) \
  ,np.mean(vrott_sam,axis=0).reshape(ntrail) \
  ,xerr=np.std(uradt_sam,axis=0).reshape(ntrail) \
  ,yerr=np.std(vrott_sam,axis=0).reshape(ntrail),fmt='.')
plt.xlabel(r"U (km/s)",fontsize=12,fontname="serif")
plt.ylabel(r"V (km/s)",fontsize=12,fontname="serif")
plt.axis([-60.0,60.0,-60.0,60.0],'scaled')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plot
plt.tight_layout()
plt.show()


##### plot

# set radial velocity arrow
i=0
vxarr=np.zeros((2,nstarv))
vyarr=np.zeros((2,nstarv))
vxarr[0,:]=xposv
vyarr[0,:]=yposv
# set dx dy
vxarr[1,:]=vxv/100.0
vyarr[1,:]=vyv/100.0

# plot circle, 
an=np.linspace(0,2.0*np.pi,100)
rad=7.0
i=0
rad=4.0
while i<15:
  rad=rad+0.5
  plt.plot(rad*np.cos(an),rad*np.sin(an),'k:')
  i+=1
# plot arm position from Reid et al. 2014
# number of points
nsp=100
isp=0
plotsparm=True
if plotsparm==True:
  numsp=3
else:
  numsp=0
while isp<numsp:
# angle in R14 is clock-wise start at the Sun at (0.0, Rsun)
# convert to the one anti-clockwise starting from +x, y=0
  if isp==0:
# Scutum Arm  
    angen=(180.0-3.0)*np.pi/180.0
#    angen=(180.0+45.0)*np.pi/180.0
    angst=(180.0-101.0)*np.pi/180.0
    angref=(180.0-27.6)*np.pi/180.0
    rref=5.0
# pitchangle
    tanpa=np.tan(19.8*np.pi/180.0)
  elif isp==1:
# Sagittarius Arm  
    angen=(180.0+2.0)*np.pi/180.0
#    angen=(180.0+45.0)*np.pi/180.0
    angst=(180.0-68.0)*np.pi/180.0
    angref=(180.0-25.6)*np.pi/180.0
    rref=6.6
# pitchangle
    tanpa=np.tan(6.9*np.pi/180.0)
  else:
# Perseus Arm  
    angen=(180.0-88.0)*np.pi/180.0
    angst=(180.0+21.0)*np.pi/180.0
    angref=(180.0-14.2)*np.pi/180.0
    rref=9.9
# pitchangle
    tanpa=np.tan(9.4*np.pi/180.0)
# logarithmic spiral arm , log r= tan(pa) theta, in the case of anti-clockwise arm
  an=np.linspace(angst,angen,nsp)
  xsp=np.zeros(nsp)
  ysp=np.zeros(nsp)
  i=0
  while i<nsp:
    rsp=np.exp(tanpa*(an[i]-angref))*rref
    xsp[i]=rsp*np.cos(an[i])
    ysp[i]=rsp*np.sin(an[i])
    i+=1
  if isp==0:
    plt.plot(xsp,ysp,'b-')
  elif isp==1:
    plt.plot(xsp,ysp,'r-')
  else:
    plt.plot(xsp,ysp,'g-')
    f=open('PerseusArm.asc','w')
    for i in range(nsp):
      print >>f,"%f %f" %(xsp[i],ysp[i])
    f.close()
  isp+=1

# velocity arrow 
i=0
while i<nstarv:
# x,y,dx,dy
  plt.arrow(vxarr[0,i],vyarr[0,i],vxarr[1,i],vyarr[1,i]
  ,fc="k", ec="k",head_width=0.05, head_length=0.1 )
  i+=1

# plot Cepheids data point
plt.scatter(-rsun,0.0,marker="*",s=100,color='k')
plt.scatter(xposv,yposv,c=darmv,s=30,vmin=-4.0,vmax=4.0)
plt.xlabel(r"X (kpc)",fontsize=18,fontname="serif")
plt.ylabel(r"Y (kpc)",fontsize=18,fontname="serif")
plt.axis([-13.0,-3.0,-4.5,4.5],'scaled')
cbar=plt.colorbar()
cbar.set_label(r'$\delta$[Fe/H]')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()


