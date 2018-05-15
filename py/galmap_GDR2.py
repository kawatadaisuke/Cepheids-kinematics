
#
#  Read Cepheids/Genovali14/G14xGDRd1xM15.fits and IYCep-combinedxM15.fits
#
#  12/05/2018  Written - Daisuke Kawata
#

import pyfits
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from scipy import stats
from galpy.util import bovy_coords

# Galactic parameters
# constant for proper motion unit conversion
pmvconst = 4.74047
# from Bland-Hawthorn & Gerhard
usun = 10.0
vsun = 248.0
wsun = 7.0
vcircsun = 248.0-11.0
rsun = 8.2
zsun = 0.025

nfile = 2
glon = np.array([])
glat = np.array([])
ra = np.array([])
dec = np.array([])
plx = np.array([])
e_plx = np.array([])
mod = np.array([])
e_mod = np.array([])
pmra = np.array([])
pmdec = np.array([])
e_pmra = np.array([])
e_pmdec = np.array([])
hrv_rvs = np.array([])
hrv_m15 = np.array([])
e_hrv_rvs = np.array([])
e_hrv_m15 = np.array([])
n_hrv_rvs = np.array([])
logp = np.array([])
vflag = np.array([])
gmag = np.array([])

for ii in range(nfile):
    # input parameters
    # input data
    if ii == 0:
        infile='/Users/dkawata/work/obs/Cepheids/Genovali14/G14xGDR2d1xM15.fits'
    else:
        infile='/Users/dkawata/work/obs/Cepheids/Genovali14/IYCep-combinedxM15.fits'
    star_hdus=pyfits.open(infile)
    star=star_hdus[1].data
    star_hdus.close()

    # read the data
    # number of data points
    nstar=len(star['mod'])
    print infile,' number of stars =',nstar

    sindx = np.where((star['e_HRV']>0.0))
    # sindx = np.where((star['parallax_error']>0.0))

    # extract the necessary particle info
    # use Gaia position 
    glon = np.append(glon, star['l'][sindx])
    glat = np.append(glat, star['b'][sindx])
    ra = np.append(ra, star['ra'][sindx])
    dec = np.append(dec, star['dec'][sindx])
    # plx = np.append(plx, star['parallax'][sindx])+0.029
    plx = np.append(plx, star['parallax'][sindx])
    e_plx = np.append(e_plx, star['parallax_error'][sindx])
    mod = np.append(mod, star['mod'][sindx])
    e_mod = np.append(e_mod, star['e_mod'][sindx])
    pmra =np.append(pmra, star['pmra'][sindx])
    pmdec = np.append(pmdec, star['pmdec'][sindx])
    e_pmra = np.append(e_pmra, star['pmra_error'][sindx])
    e_pmdec = np.append(e_pmdec, star['pmdec_error'][sindx])
    hrv_rvs = np.append(hrv_rvs, star['radial_velocity'][sindx])
    hrv_m15 = np.append(hrv_m15, star['HRV'][sindx])
    e_hrv_rvs = np.append(e_hrv_rvs, star['radial_velocity_error'][sindx])
    e_hrv_m15 = np.append(e_hrv_m15, star['e_HRV'][sindx])
    n_hrv_rvs = np.append(n_hrv_rvs, star['rv_nb_transits'][sindx])
    logp = np.append(logp, star['logper'][sindx])
    vflag = np.append(vflag, star['phot_variable_flag'][sindx])
    gmag = np.append(gmag, star['phot_g_mean_mag'][sindx])

print ' Total number of stars =',len(hrv_m15)

# distance comparison
dist_mod=np.power(10.0,(mod+5.0)/5.0)*0.001
e_dist_mod=np.zeros([2,len(dist_mod)])
e_dist_mod[0,:]=np.fabs(np.power(10.0,(mod+e_mod+5.0)/5.0)*0.001-dist_mod)
e_dist_mod[1,:]=np.fabs(np.power(10.0,(mod-e_mod+5.0)/5.0)*0.001-dist_mod)
dist_plx=1.0/plx
e_dist_plx = np.zeros([2,len(dist_plx)])
e_dist_plx[0,:]=np.fabs(1.0/(plx-e_plx)-dist_plx)
e_dist_plx[1,:]=np.fabs(1.0/(plx+e_plx)-dist_plx)

distlim = 5.0
sindx = np.where((plx>0.0) & (dist_plx<distlim) & (dist_mod<distlim) & (e_hrv_m15>0.0) & (gmag>10.0))
print ' N (d<',distlim,') (kpc)',len(dist_plx[sindx])
print ' mean and std diff fraction =', \
  np.mean((dist_plx[sindx]-dist_mod[sindx])/dist_plx[sindx]), \
  np.std((dist_plx[sindx]-dist_mod[sindx])/dist_plx[sindx])
print ' mean and std parallax diff =', \
  np.mean((plx[sindx]-1.0/dist_mod[sindx])), \
  np.std((plx[sindx]-1.0/dist_mod[sindx]))

distlim = 4.0
sindx = np.where((plx>0.0) & (dist_plx<distlim) & (dist_mod<distlim) & (e_hrv_m15>0.0))
print ' N (d<',distlim,') (kpc)',len(dist_plx[sindx])
print ' mean and std diff =', \
  np.mean((dist_plx[sindx]-dist_mod[sindx])/dist_plx[sindx]), \
  np.std((dist_plx[sindx]-dist_mod[sindx])/dist_plx[sindx])
print ' mean and std diff =', \
  np.mean((plx[sindx]-1.0/dist_mod[sindx])), \
  np.std((plx[sindx]-1.0/dist_mod[sindx]))

f = open('dist_comp.asc', 'w')
for ii in range(len(dist_mod)):
    print >> f,"%f %f %f %f %f %f %f %f %f %f" % ( \
        dist_plx[ii], e_dist_plx[0, ii], e_dist_plx[1, ii], \
        dist_mod[ii], e_dist_mod[0, ii], e_dist_mod[1, ii], \
        plx[ii], e_plx[ii], mod[ii], \
        e_mod[ii])
f.close()

### distance comparison.
xrange = np.array([0.0,2.0])
yrange = np.array([0.0,2.0])
plt.errorbar(dist_plx, dist_mod, xerr=e_dist_plx, yerr=e_dist_mod, fmt='.'  \
    , marker='.')
# x=y line
nsp=2
xsp=np.linspace(xrange[0], xrange[1], nsp)
ysp=xsp
plt.plot(xsp,ysp,'b-')
plt.xlabel(r"D(parallax) (kpc)",fontsize=18,fontname="serif")
plt.ylabel(r"D(Mod)",fontsize=18,fontname="serif")
plt.axis([xrange[0], xrange[1], yrange[0], yrange[1]],'scaled')
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
plt.show()

# RVS vs. M15
sindx = np.where((e_hrv_rvs>0.0) & (e_hrv_m15>0.0))
print ' N HRV with RVS and M15=',len(hrv_rvs[sindx])
print ' HRV diff mean, std=',np.mean(hrv_rvs[sindx]-hrv_m15[sindx]), \
   np.std(hrv_rvs[sindx]-hrv_m15[sindx])
sindx = np.where((e_hrv_rvs>0.0) & (e_hrv_m15>0.0) & (n_hrv_rvs>=5))
print ' N (N_RVS>=5) with RVS and M15=',len(hrv_rvs[sindx])
print ' HRV diff mean, std=',np.mean(hrv_rvs[sindx]-hrv_m15[sindx]), \
   np.std(hrv_rvs[sindx]-hrv_m15[sindx])
xrange = np.array([-150.0,150.0])
yrange = np.array([-150.0,150.0])
plt.errorbar(hrv_rvs, hrv_m15, xerr=e_hrv_rvs, yerr=e_hrv_m15, fmt='.'  \
    , marker='.')
# x=y line
nsp=2
xsp=np.linspace(xrange[0], xrange[1], nsp)
ysp=xsp
plt.plot(xsp,ysp,'b-')
plt.xlabel(r"Vlos (RVS) (km/s)",fontsize=18,fontname="serif")
plt.ylabel(r"Vlos (M15) (km/s)",fontsize=18,fontname="serif")
plt.axis([xrange[0], xrange[1], yrange[0], yrange[1]],'scaled')
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
plt.show()

# RVS-M15 vs. Log P
sindx = np.where((e_hrv_rvs>0.0) & (e_hrv_m15>0.0) & (logp<1.1))
print ' Log P<1.1, HRV-M15/error diff mean, std=', \
   np.mean((hrv_rvs[sindx]-hrv_m15[sindx])/np.sqrt(e_hrv_rvs[sindx]**2+e_hrv_m15[sindx]**2)), \
   np.std((hrv_rvs[sindx]-hrv_m15[sindx])/np.sqrt(e_hrv_rvs[sindx]**2+e_hrv_m15[sindx]**2))

sindx = np.where((e_hrv_rvs>0.0) & (e_hrv_m15>0.0))
print ' All common HRV-M15/error diff mean, std=', \
   np.mean((hrv_rvs[sindx]-hrv_m15[sindx])/np.sqrt(e_hrv_rvs[sindx]**2+e_hrv_m15[sindx]**2)), \
   np.std((hrv_rvs[sindx]-hrv_m15[sindx])/np.sqrt(e_hrv_rvs[sindx]**2+e_hrv_m15[sindx]**2))

xrange = np.array([0.0, 2.0])
yrange = np.array([-30.0,50.0])
dverr = np.sqrt(e_hrv_rvs[sindx]**2+e_hrv_m15[sindx]**2)
plt.errorbar(logp[sindx], hrv_rvs[sindx]-hrv_m15[sindx], yerr=dverr, fmt='.',  \
    marker='.')
plt.xlabel(r"Log P (days)",fontsize=18,fontname="serif")
plt.ylabel(r"Vlos(RVS)-Vlos(M15) (km/s)",fontsize=18,fontname="serif")
plt.axis([xrange[0], xrange[1], yrange[0], yrange[1]],'scaled')
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
plt.show()

### finally selected stars
sindx = np.where((e_hrv_m15>0.0) & (dist_mod<4.0))
# sindx = np.where((e_hrv_rvs>0.0) & (dist_mod<4.0))
print ' final selection of stars =',len(glon[sindx])
# degree to radian
glonrads = glon[sindx]*np.pi/180.0
glatrads = glat[sindx]*np.pi/180.0
# distance x,y position for selected stars
dists = dist_mod[sindx]
xpos = -rsun+dists*np.cos(glonrads)*np.cos(glatrads)
ypos = dists*np.sin(glonrads)*np.cos(glatrads)

# velocity conversion
# convert proper motion from mu_alpha,delta to mu_l,b using bovy_coords
Tpmllbb=bovy_coords.pmrapmdec_to_pmllpmbb(pmra[sindx], pmdec[sindx], ra[sindx], dec[sindx], degree=True, epoch=None)
# use galpy
# print ' shape=',np.shape(hrv_m15[sindx]),np.shape(Tpmllbb),np.shape(glon[sindx]), np.shape(glat[sindx]), np.shape(dists)
Tvxvyvz=bovy_coords.vrpmllpmbb_to_vxvyvz(hrv_m15[sindx], Tpmllbb[:,0], \
    Tpmllbb[:,1], glon[sindx], glat[sindx], dists, degree=True)
# use RVS data
# Tvxvyvz=bovy_coords.vrpmllpmbb_to_vxvyvz(hrv_rvs[sindx], Tpmllbb[:,0], \
#    Tpmllbb[:,1], glon[sindx], glat[sindx], dists, degree=True)
vxs=Tvxvyvz[:,0]+usun
vys=Tvxvyvz[:,1]+vsun
vzs=Tvxvyvz[:,2]+zsun
# Galactic radius and velocities
rgals=np.sqrt(xpos**2+ypos**2)
# then subtract circular velocity contribution
vxs=vxs-vcircsun*ypos/rgals
vys=vys+vcircsun*xpos/rgals
vrads=(vxs*xpos+vys*ypos)/rgals
vrots=(vxs*ypos-vys*xpos)/rgals

# set velocity arrows
vxarr = np.zeros((2, len(vxs)))
vyarr = np.zeros((2, len(vxs)))
vxarr[0,:] = xpos
vyarr[0,:] = ypos
vxarr[1,:] = vxs/100.0
vyarr[1,:] = vys/100.0

f = open('posv_g14gdr2m15.asc', 'w')
for ii in range(len(xpos)):
    print >> f,"%f %f %f %f" % ( \
        xpos[ii], ypos[ii], vxs[ii], vys[ii])
f.close()

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
# velocity arrow
for i in range(len(xpos)):
# x,y,dx,dy
    plt.arrow(vxarr[0,i],vyarr[0,i],vxarr[1,i],vyarr[1,i] \
        ,fc="k", ec="k",head_width=0.05, head_length=0.1 )
plt.scatter(-rsun,0.0,marker="*",s=100,color='k')
plt.scatter(xpos,ypos)
plt.xlabel(r"X (kpc)",fontsize=18,fontname="serif")
plt.ylabel(r"Y (kpc)",fontsize=18,fontname="serif")
plt.axis([-13.0,-3.0,-4.5,4.5],'scaled')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

