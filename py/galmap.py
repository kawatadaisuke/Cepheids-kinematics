
# 1. Read ../table3+4.fits from Genovali et al. (2014)
# http://vizier.nao.ac.jp/viz-bin/VizieR-3?-source=J/A%2bA/566/A37/table4
#  analyse the metallicity gradient
# 2. Read 
# ../G14T35+TGAS+Melnik15-Gorynya.fits which is the cross-mathed data of
# Tables 3 and 4 of Genovali et al. (2014) metallicity
# TGAS for proper motion
# ../../Melnik15
#  minus sample in G14T34+TGAS+Gorynya.fits 
# 3. Read
#  ../G14T34+TGAS+DDO16-Melnik15-Gorynya.fits
#  ../../DDO16 sample - Gorynya - Melnik15
#
# Gorynya 1992-98, from http://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=III/229&-out.max=50&-out.form=HTML%20Table&-out.add=_r&-out.add=_RAJ,_DEJ&-sort=_r&-oc.form=sexa
# 4. Plot x-y distribution and the arros of peculiar velocity
# 
# History:
#  23/06/2017 move to obs/projs/Cepheids-kinematics/py
#  22/06/2017 add velocity dispersion profile
#  19/06/2017 ver.3: reading 3 files for G14, Melnik15, DDO16
#  13/03/2017 ver.2: reading both G14T35+TGAS+DDO16.fits and G14T34+TGAS+Gorynya-DDO.fits - Daisuke Kawata
#  05/03/2017  Written - Daisuke Kawata 
# 

import pyfits
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from scipy import stats
from galpy.util import bovy_coords

# input parameters
# input data
infile='/Users/dkawata/work/obs/Cepheids/Genovali14/table3+4.fits'
star_hdus=pyfits.open(infile)
star=star_hdus[1].data
star_hdus.close()

# read the data
# number of data points
nstar=len(star['Mod'])
print 'number of particles=',nstar

# extract the necessary particle info
glon=star['_Glon']
glat=star['_Glat']
# rescaled Fe/H
feh=star['__Fe_H_']
dist=np.power(10.0,(star['Mod']+5.0)/5.0)*0.001
rgalg14=star['Rgal']*0.001

# circular velocity 
# Reid et al. (2014)
vcirc=240.0
# Sun's radius used in Reid et al. (2014)
xsunr14=-8.34
#
# Sun's proper motion Schoenrich et al.
usun=11.1
vsun=12.24
zsun=7.25
# Bovylev (2017)
# usun=7.9
# vsun=11.73
# degree to radian
glonrad=glon*np.pi/180.0
glatrad=glat*np.pi/180.0
# x,y position
xpos=xsunr14+np.cos(glonrad)*dist*np.cos(glatrad)
ypos=np.sin(glonrad)*dist*np.cos(glatrad)
# rgal with Reid et al. value
rgal=np.sqrt(xpos**2+ypos**2)

# linear regression of metallicity gradient
slope, intercept, r_value, p_value, std_err = stats.linregress(rgal,feh)
print ' slope, intercept=',slope,intercept

# delta feh
delfeh=feh-(slope*rgal+intercept)


# output ascii data for test
f=open('cepheidsg14pos.asc','w')
i=0
while i < nstar:
  print >>f, "%f %f %f %f %f %f" %(xpos[i],ypos[i],rgalg14[i],rgal[i],feh[i],delfeh[i])
  i+=1
f.close()

# output fits file
tbhdu = pyfits.BinTableHDU.from_columns([\
  pyfits.Column(name='X',format='D',array=xpos),\
  pyfits.Column(name='Y',format='D',array=ypos),\
  pyfits.Column(name='d[Fe/H]',format='D',array=delfeh),\
  pyfits.Column(name='Rgal',format='D',array=rgal),\
  pyfits.Column(name='[Fe/H]',format='D',array=feh)])
tbhdu.writeto('cepheidsg14posdfeh.fits',clobber=True)

### plot radial metallicity distribution
# plot Cepheids data point
plt.scatter(rgal,feh,c=delfeh,s=30,vmin=-0.1,vmax=0.25)
# radial gradient
nsp=10
xsp=np.linspace(4.0,20.0,nsp)
ysp=slope*xsp+intercept
plt.plot(xsp,ysp,'b-')
plt.xlabel(r"R (kpc)",fontsize=18,fontname="serif")
plt.ylabel(r"[Fe/H]",fontsize=18,fontname="serif")
plt.axis([4.0,20.0,-1.0,0.75],'scaled')
cbar=plt.colorbar()
cbar.set_label(r'$\delta$[Fe/H]')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

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
numsp=3
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
  isp+=1

# plot Cepheids data point
plt.scatter(xsunr14,0.0,marker="*",s=100,color='k')
plt.scatter(xpos,ypos,c=delfeh,s=30,vmin=-0.1,vmax=0.25)
plt.xlabel(r"X (kpc)",fontsize=18,fontname="serif")
plt.ylabel(r"Y (kpc)",fontsize=18,fontname="serif")
plt.axis([-13.0,-3.0,-4.5,4.5],'scaled')
cbar=plt.colorbar()
cbar.set_label(r'$\delta$[Fe/H]')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()


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
errhrvv=np.hstack((errhrvv,np.zeros(nstarv3)))
logp=np.hstack((logp,star['logPer']))

# use galpy RA,DEC -> Glon,Glat
# Tlb=bovy_coords.radec_to_lb(rav,decv,degree=True,epoch=2000.0)
# degree to radian
glonradv=glonv*np.pi/180.0
glatradv=glatv*np.pi/180.0
# x,y position
xposv=xsunr14+np.cos(glonradv)*distv*np.cos(glatradv)
yposv=np.sin(glonradv)*distv*np.cos(glatradv)
# rgal with Reid et al. value
rgalv=np.sqrt(xposv**2+yposv**2)
delfehv =fehv-(slope*rgalv+intercept)
pmvconst=4.74047

# convert proper motion to vx,vy with my ugry conversion

# get Cartician position in equatorial coordidate
# distance in kpc etc. alphao,deltao (radian)
disto=distv
alphao=rav*np.pi/180.0
deltao=decv*np.pi/180.0
# muao with true arc
muao=pmrav
mudo=pmdecv
vrado=hrvv
# x,y,z position from the sun's position
rxy=disto*np.cos(deltao)
px=rxy*np.cos(alphao)
py=rxy*np.sin(alphao)
pz=disto*np.sin(deltao)
# vx,vy,vz
# mas/yr -> km/s
valpha=pmvconst*muao*distv
vdelta=pmvconst*mudo*distv
vxy=vrado*np.cos(deltao)-vdelta*np.sin(deltao)
vx=vxy*np.cos(alphao)-valpha*np.sin(alphao)
vy=vxy*np.sin(alphao)+valpha*np.cos(alphao)
vz=vrado*np.sin(deltao)+vdelta*np.cos(deltao)

# convert to galactic coordinate
# 3 axes in Galactic coordinate
# North Celestial Pole (z-axis)
lzeq=np.radians(122.93193212)
bzeq=np.radians(27.12835496)
# RA,DEC=0,0 (x-axis)
lxeq=np.radians(96.33723825)
bxeq=np.radians(-60.18853909)
#  RA,DEC=90,0 (y-axis)
lyeq=np.radians(206.98916373)
byeq=np.radians(-11.42442440)
# transformation matrix (though it is array in python)
tmateqga=np.array([
  [np.cos(lxeq)*np.cos(bxeq),np.sin(lxeq)*np.cos(bxeq),np.sin(bxeq)],
  [np.cos(lyeq)*np.cos(byeq),np.sin(lyeq)*np.cos(byeq),np.sin(byeq)],
 [np.cos(lzeq)*np.cos(bzeq),np.sin(lzeq)*np.cos(bzeq),np.sin(bzeq)]
])
print 'tmat gal->eq=',tmateqga
# take inverse
tmatgaeq=np.linalg.inv(tmateqga)
print 'tmat eq->gal=',tmatgaeq

print 'tmat gal->eq= %12.9f %12.9f %12.9f' % (tmateqga[0,0],tmateqga[0,1],tmateqga[0,2])
print '              %12.9f %12.9f %12.9f' % (tmateqga[1,0],tmateqga[1,1],tmateqga[1,2])
print '              %12.9f %12.9f %12.9f' % (tmateqga[2,0],tmateqga[2,1],tmateqga[2,2])

print 'tmat eq->gal= %12.9f %12.9f %12.9f' % (tmatgaeq[0,0],tmatgaeq[0,1],tmatgaeq[0,2])
print '              %12.9f %12.9f %12.9f' % (tmatgaeq[1,0],tmatgaeq[1,1],tmatgaeq[1,2])
print '              %12.9f %12.9f %12.9f' % (tmatgaeq[2,0],tmatgaeq[2,1],tmatgaeq[2,2])

# transfer the coordinate: x-y plane: Cerestial equator -> Galactic plane
# position
# position in Galactic cartesian coordinate
poseqs=np.vstack([px,py,pz])
# transfer to equatorial cartesian coordinate
posgas=np.dot(tmatgaeq,poseqs)

# velocity
# velocity in Galactic cartesian coordinate
veleqs=np.vstack([vx,vy,vz])
# transfer to equatorial cartesian coordinate
velgas=np.dot(tmatgaeq,veleqs)

# reset position and velocity
px=posgas[0,:]+xsunr14
py=posgas[1,:]
pz=posgas[2,:]
vx=velgas[0,:]+usun
vy=velgas[1,:]+vsun
vz=velgas[2,:]

rad=np.sqrt(np.power(px,2)+np.power(py,2))

print ' rmax=',np.max(rad)

# rotation velocity 
vrot=(vx*py-vy*px)/rad
# radial velocity 
vrad=(vx*px+vy*py)/rad

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
# vxv=hrvv*np.cos(glonradv)*np.cos(glatradv)-vlonv*np.sin(glonradv) \
#  -vlatxyv*np.cos(glonradv)+usun
# vyv=hrvv*np.sin(glonradv)*np.cos(glatradv)+vlonv*np.cos(glonradv) \
#  -vlatxyv*np.sin(glonradv)+vsun
# use galpy
Tvxvyvz=bovy_coords.vrpmllpmbb_to_vxvyvz(hrvv,pmlonv,pmlatv,glonv,glatv,distv,degree=True)
vxv=Tvxvyvz[:,0]+usun
vyv=Tvxvyvz[:,1]+vsun
vzv=Tvxvyvz[:,2]+zsun
vyv=vyv+vcirc
# original velocity
vxv0=vxv
vyv0=vyv
vzv0=vzv
# Galactic radius and velocities
rgalv=np.sqrt(xposv**2+yposv**2)
vradv0=(vxv0*xposv+vyv0*yposv)/rgalv
vrotv0=(vxv0*yposv-vyv0*xposv)/rgalv
# then subtract circular velocity contribution
vxv=vxv-vcirc*yposv/rgalv
vyv=vyv+vcirc*xposv/rgalv
vradv=(vxv*xposv+vyv*yposv)/rgalv
vrotv=(vxv*yposv-vyv*xposv)/rgalv

# output fits
# Fits output
# output fits file
tbhdu = pyfits.BinTableHDU.from_columns([\
  pyfits.Column(name='Name',format='A20',array=name),\
  pyfits.Column(name='X',format='D',array=xposv),\
  pyfits.Column(name='Y',format='D',array=yposv),\
  pyfits.Column(name='Vx',format='D',array=vxv0),\
  pyfits.Column(name='Vy',format='D',array=vyv0),\
  pyfits.Column(name='[Fe/H]',format='D',array=fehv),\
  pyfits.Column(name='d[Fe/H]',format='D',array=delfehv),\
  pyfits.Column(name='Rgal',format='D',array=rgalv),\
  pyfits.Column(name='Vrad',format='D',array=vradv0),\
  pyfits.Column(name='Vrot',format='D',array=vrotv0),\
  pyfits.Column(name='Dist',format='D',array=distv), \
  pyfits.Column(name='delVx',format='D',array=vxv), \
  pyfits.Column(name='delVy',format='D',array=vyv), \
  pyfits.Column(name='delVrad',format='D',array=vradv), \
  pyfits.Column(name='delVrot',format='D',array=vrotv), \
  pyfits.Column(name='Glon',format='D',array=glonv), \
  pyfits.Column(name='Glat',format='D',array=glatv), \
  pyfits.Column(name='RA',format='D',array=rav), \
  pyfits.Column(name='DEC',format='D',array=decv), \
  pyfits.Column(name='PMRA',format='D',array=pmrav), \
  pyfits.Column(name='e_PMRA',format='D',array=errpmrav), \
  pyfits.Column(name='PMDEC',format='D',array=pmdecv), \
  pyfits.Column(name='e_PMDEC',format='D',array=errpmdecv), \
  pyfits.Column(name='HRV',format='D',array=hrvv), \
  pyfits.Column(name='e_HRV',format='D',array=errhrvv), \
  pyfits.Column(name='LogPer',format='D',array=logp)])
tbhdu.writeto('cepheidspv.fits',clobber=True)

# output ascii data to check consistency between two conversions. 
# checked on 05/02/2017
f=open('comps.asc','w')
i=0
while i<nstarv:
  print >>f,"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f" \
    %(xposv[i],yposv[i] \
      ,vxv[i],vyv[i],px[i],py[i],vx[i],vy[i],distv[i],rav[i],decv[i] \
      ,pmrav[i],pmdecv[i],hrvv[i],glonv[i],glatv[i])
  i+=1
f.close()

# check proper motion velocity errors
errpmrav=pmvconst*distv*errpmrav
errpmdecv=pmvconst*distv*errpmdecv
sindx=np.where(np.logical_and(errhrvv>0.0,np.logical_and(errhrvv<5.0,np.logical_and(errpmrav<5.0,errpmdecv<5.0))))
namepme=name[sindx]
xposvpme=xposv[sindx]
yposvpme=yposv[sindx]
delfehvpme=delfehv[sindx]
vxvpme=vxv[sindx]
vyvpme=vyv[sindx]
distpme=distv[sindx]
vlonpme=vlonv[sindx]
nspme=len(vyvpme)
print 'proper motion error < 5 km/s nspme=',nspme
# Galactic radius and velocities
rgalpme=np.sqrt(xposvpme**2+yposvpme**2)
vradpme=(vxvpme*xposvpme+vyvpme*yposvpme)/rgalpme
vrotpme=(vxvpme*yposvpme-vyvpme*xposvpme)/rgalpme
vrad0pme=vradv0[sindx]
vrot0pme=vrotv0[sindx]
vz0pme=vzv0[sindx]

# output ascii data for test
f=open('cepheidspv.asc','w')
i=0
for i in range(nspme):
  print >>f,"%f %f %f %f %f %f %f %f %f %f" %(xposvpme[i],yposvpme[i] \
    ,vxvpme[i],vyvpme[i],delfehvpme[i],rgalpme[i],vradpme[i] \
    ,vrotpme[i],distpme[i],vlonpme[i])
f.close()

# set radial velocity arrow
i=0
vxarr=np.zeros((2,nspme))
vyarr=np.zeros((2,nspme))
vxarr[0,:]=xposvpme
vyarr[0,:]=yposvpme
# set dx dy
vxarr[1,:]=vxvpme/100.0
vyarr[1,:]=vyvpme/100.0

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
numsp=3
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

# wild guess of extension
isp=0
numsp=2
while isp<numsp:
# angle in R14 is clock-wise start at the Sun at (0.0, Rsun)
# convert to the one anti-clockwise starting from +x, y=0
  if isp==0:
# Scutum Arm  
# pitchangle
    tanpa=np.tan(19.8*np.pi/180.0)
    rref=5.0*np.exp(tanpa*((-3.0+27.6)*np.pi/180.0))
    angst=(180.0-3.0)*np.pi/180.0
    angen=(180.0+45.0)*np.pi/180.0
    angref=angst
# extended pitchangle
    tanpa=np.tan(5.0*np.pi/180.0)
  elif isp==1:
# Sagittarius Arm  
# pitchangle
    tanpa=np.tan(6.9*np.pi/180.0)
    rref=6.6*np.exp(tanpa*((+2.0+25.6)*np.pi/180.0))
    angst=(180.0+2.0)*np.pi/180.0
    angen=(180.0+45.0)*np.pi/180.0
    angref=angst
# extended pitchangle
    tanpa=np.tan(15.0*np.pi/180.0)
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
    plt.plot(xsp,ysp,'b--')
  else:
    plt.plot(xsp,ysp,'r--')
  isp+=1

# plot tangent 
dist=6.0
isp=0
nsp=2
xline=np.zeros(2)
yline=np.zeros(2)
xline[0]=xsunr14
yline[0]=0.0
while isp<nsp:
  if isp==0:
# Crux Tangent
    tang=310.0
  else:
# Carina Tangent
    tang=284.0
  xline[1]=xsunr14+dist*np.cos(tang*np.pi/180.0)
  yline[1]=dist*np.sin(tang*np.pi/180.0)
  if isp==0:
    plt.plot(xline,yline,'b--')
  else:
    plt.plot(xline,yline,'r--')
  isp+=1

# velocity arrow 
i=0
while i<nspme:
# x,y,dx,dy
  plt.arrow(vxarr[0,i],vyarr[0,i],vxarr[1,i],vyarr[1,i]
  ,fc="k", ec="k",head_width=0.05, head_length=0.1 )
  i+=1

# plot Cepheids data point
plt.scatter(xsunr14,0.0,marker="*",s=100,color='k')
plt.scatter(xposvpme,yposvpme,c=delfehvpme,s=30,vmin=-0.1,vmax=0.25)
plt.xlabel(r"X (kpc)",fontsize=18,fontname="serif")
plt.ylabel(r"Y (kpc)",fontsize=18,fontname="serif")
plt.axis([-13.0,-3.0,-4.5,4.5],'scaled')
cbar=plt.colorbar()
cbar.set_label(r'$\delta$[Fe/H]')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# velocity dispersion for all
print ' Vmean,rot,rad,z=',np.mean(vrot0pme),np.mean(vrad0pme),np.mean(vz0pme)
print 'Vsig,rot,rad,z=',np.std(vrot0pme),np.std(vrad0pme),np.std(vz0pme)

# velocity dispersion analysis
nrbin=3
rmin=6.0
rmax=12.0
print ' nrbin,rmin,rmax=',nrbin,rmin,rmax
# number of stars
np_r=np.histogram(rgalpme,nrbin,(rmin,rmax))[0]
print 'number of stars =',np_r
# mean radius
rmean_r=np.histogram(rgalpme,nrbin,(rmin,rmax),weights=rgalpme)[0]/np_r

# rotation velocity
# mean
vrotm_r=np.histogram(rgalpme,nrbin,(rmin,rmax),weights=vrot0pme)[0]/np_r
# square mean
vrot2m_r=np.histogram(rgalpme,nrbin,(rmin,rmax),weights=vrot0pme**2)[0]/np_r
# velocity dispersion
vrotsig_r=np.sqrt(vrot2m_r-vrotm_r**2)
print ' Vrot,mean=',vrotm_r
print ' Vsig,rot=',vrotsig_r

# radial velocity
# mean
vradm_r=np.histogram(rgalpme,nrbin,(rmin,rmax),weights=vrad0pme)[0]/np_r
# square mean
vrad2m_r=np.histogram(rgalpme,nrbin,(rmin,rmax),weights=vrad0pme**2)[0]/np_r
# velocity dispersion
vradsig_r=np.sqrt(vrad2m_r-vradm_r**2)
print ' Vrad,mean=',vradm_r
print ' Vsig,rad=',vradsig_r

# vertical velcoity 
vzm_r=np.histogram(rgalpme,nrbin,(rmin,rmax),weights=vz0pme)[0]/np_r
# square mean
vz2m_r=np.histogram(rgalpme,nrbin,(rmin,rmax),weights=vz0pme**2)[0]/np_r
# velocity dispersion
vzsig_r=np.sqrt(vz2m_r-vzm_r**2)
print ' Vz,mean=',vzm_r
print ' Vsig,z=',vzsig_r

rminplot=5.0001
rmaxplot=11.999
gs1=gridspec.GridSpec(3,1)
gs1.update(left=0.15,right=0.9,bottom=0.1,top=0.95,hspace=0,wspace=0)

# Vrot 
plt.subplot(gs1[0])
# labes
plt.ylabel(r"$\rm V_{rot}$",fontsize=18,fontname="serif",style="normal")
# scatter plot
plt.scatter(rgalpme,vrot0pme,c=delfehvpme,s=30,vmin=-0.25,vmax=0.25)
# hexbin plot
# plt.hexbin(rgalpme,vrot0pme,bins='log',gridsize=300,cmap=cm.jet)
# plot mean
plt.errorbar(rmean_r,vrotm_r,yerr=vrotsig_r,fmt='ok')
plt.axis([rminplot,rmaxplot,200.0,300.0])
cbar=plt.colorbar()
cbar.set_label(r'$\delta$[Fe/H]')

# Vrad
plt.subplot(gs1[1])
# labes
plt.ylabel(r"$\rm V_{rad}$",fontsize=18,fontname="serif",style="normal")
# scatter plot
plt.scatter(rgalpme,vrad0pme,c=delfehvpme,s=30,vmin=-0.25,vmax=0.25)
# plot mean and dispersion
plt.errorbar(rmean_r,vradm_r,yerr=vradsig_r,fmt='ok')
plt.axis([rminplot,rmaxplot,-50.0,50.0])
cbar=plt.colorbar()
cbar.set_label(r'$\delta$[Fe/H]')

# Vz
plt.subplot(gs1[2])
# labes
plt.ylabel(r"$\rm V_{z}$",fontsize=18,fontname="serif",style="normal")
# scatter plot
plt.scatter(rgalpme,vz0pme,c=delfehvpme,s=30,vmin=-0.25,vmax=0.25)
# plot mean and dispersion
plt.errorbar(rmean_r,vzm_r,yerr=vzsig_r,fmt='ok')
plt.axis([rminplot,rmaxplot,-50.0,50.0])
plt.xlabel("R (kpc)",fontsize=18,fontname="serif")
cbar=plt.colorbar()
cbar.set_label(r'$\delta$[Fe/H]')



plt.show()
