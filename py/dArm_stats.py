
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

# computer distances from the arm
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
print ' Teading vertex deviation=',(180.0/np.pi)*0.5*np.arctan(2.0*vcovt[0,1] \
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


