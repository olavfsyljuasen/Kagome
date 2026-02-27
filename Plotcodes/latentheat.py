#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

#import analyzecrossings as ac

import plotstyle

from scipy.optimize import curve_fit

from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

def f_model(x,a,b,c):
#    return a*x*(1+b*np.log(x))+c*x*x
    return a*x*(1+b*np.log(x))+c*0

#def f_model(x,a,b,c):
#    return a*x*(1+b*0)+c*x*x


def custom_formatter0(x,_):
    if x == 0:
        return '0'
    s = f'{x:.0f}'
    if s.startswith('-0'):
        return '-' + s[2:]  # Keep the minus, remove the zero
    elif s.startswith('0'):
        return s[1:]        # Remove the leading zero for positives
    else:
        return s            # For numbers >= 1 or <= -1, return as is


def custom_formatter1(x,_):
    if x == 0:
        return '0'
    s = f'{x:.1f}'
    if s.startswith('-0'):
        return '-' + s[2:]  # Keep the minus, remove the zero
    elif s.startswith('0'):
        return s[1:]        # Remove the leading zero for positives
    else:
        return s            # For numbers >= 1 or <= -1, return as is

def custom_formatter2(x,_):
    if x == 0:
        return '0'
    s = f'{x:.2f}'
    if s.startswith('-0'):
        return '-' + s[2:]  # Keep the minus, remove the zero
    elif s.startswith('0'):
        return s[1:]        # Remove the leading zero for positives
    else:
        return s            # For numbers >= 1 or <= -1, return as is

def custom_formatter3(x,_):
    if x == 0:
        return '0'
    s = f'{x:.3f}'
    if s.startswith('-0'):
        return '-' + s[2:]  # Keep the minus, remove the zero
    elif s.startswith('0'):
        return s[1:]        # Remove the leading zero for positives
    else:
        return s            # For numbers >= 1 or <= -1, return as is

    
def custom_formatter4(x,_):
    if x == 0:
        return '0'
    s = f'{x:.4f}'
    if s.startswith('-0'):
        return '-' + s[2:]  # Keep the minus, remove the zero
    elif s.startswith('0'):
        return s[1:]        # Remove the leading zero for positives
    else:
        return s            # For numbers >= 1 or <= -1, return as is



#print('backend=',plt.get_backend())
plt.ion() # interactive mode to prevent plotwindow stealing window focus  

savefigname="../Figs/"+os.path.splitext(sys.argv[0])[0]+'.pdf' #defaults to ../Figs/scriptname.pdf


fig, ax = plt.subplots(2,figsize=(6,6), layout='constrained',sharex=True)

colors=plotstyle.sortedTSpalette*3


#First discontinuities for no phonons

mydir="../Data/"

filenamelist=["latentheat.dat","tcs.dat"]

labellist  =["",""]
plotreglist=[False,False]
showdata   =[True,True]

yslist=[10000,1000]


for i in range(len(filenamelist)):

    filename=mydir+filenamelist[i]
    label = labellist[i]
    plotreg=plotreglist[i]
    ys= yslist[i]
    
    if not showdata[i]:
        continue
    
    data=pd.read_csv(filename,sep=r"\s+",header=None,names=["x","y"],comment="#")
    x=np.array(data.x)
    y=np.array(data.y)

    indices=np.where(x>20)
    x=x[indices]
    y=y[indices]
    
    x=1/x
    y=ys*np.abs(y)
    
    mplot=ax[i].plot(x, y, label=label,marker='.',markersize=10,linestyle='none')
    current_color=mplot[0].get_color()
    
    if plotreg:
        polydegree = 1
        mylinestyle = 'solid'
    else:
        polydegree = 2
        mylinestyle='dashed'

    ap= np.polyfit(x,y,polydegree)
    apf= np.poly1d(ap)
    xmin=0
    xmax=0.045
    print("y intercept=", apf[0])

    
    newx=np.linspace(xmin,xmax,100)
    ax[i].plot(newx,apf(newx),linewidth=0.5,linestyle=mylinestyle,c=current_color)



#ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter3))
#ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter2))

#ax.set_yticks([0,-0.001,-0.002,-0.003])
#ax.set_yticklabels(['0','-1','-2','-3'],fontsize=16)

ax[0].set_xticks([0,1/300,0.02,0.03,0.04])
ax[0].set_xticklabels(['0',"1/300",'.02','.03','.04'],fontsize=16)

ax[0].set_xlim(0.,0.05)
#ax.set_ylim(ys*0.0001,ys*0.000115)




ax[0].set_ylabel(r"Latent heat$[10^{-4}]$",fontsize=20)  # Add a y-label to the axes.

ax[1].set_xlabel(r'$1/L$',fontsize=20)  # Add an x-label to the axes.
ax[1].set_ylabel(r"$T_c[10^{-3}]$",fontsize=20)  # Add a y-label to the axes.

ax[0].grid(False)
ax[1].grid(False)

print('',flush=True)
plt.savefig(savefigname)

plt.pause(100000)    
#plt.show()







#plt.show()



