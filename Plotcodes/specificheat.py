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


fig, ax = plt.subplots(figsize=(6,6), layout='constrained')
axins = inset_axes(ax, 1.6, 1.6, loc='lower left', bbox_to_anchor=(0.1, 0.1),bbox_transform=ax.figure.transFigure)

colors=plotstyle.sortedTSpalette*3


#First discontinuities for no phonons

mydir="../Data/"
mydir="../Results/kagbias_2a_MERGED/0/"

#tcs:
# 8021 6 0.0023747683108167952
# 8022 9 0.0019918509317933463
# 8023 12 0.0018386672569262474
# 8024 18 0.0017152421088667679
# 8025 24 0.0016660533875579818
# 8026 30 0.0016412986587613506
# 8027 36 0.0016272474617910534
# 8028 48 0.001612503921365081
# 8029 60 0.0016054709365538238
# 8030 90 0.0015985673154950445
# 8031 120 0.0015961249040368536
# 8032 180 0.001594402809647902
# 8033 240 0.0015937506958029915
# 8034 300 0.001593489110392417


labellist=["6","9","12","18","36","120"]
#plotreglist=[False]
showdata=   [True,True,True,True,True,True]*10

#ys=10000

Tclist         =[0.0023747683108167952,0.0019918509317933463,0.0018386672569262474,0.0017152421088667679,0.0016272474617910534,0.0015961249040368536]
filenamelisthig=["r8021.tf.lowT.dat","r8022.tf.lowT.dat","r8023.tf.lowT.dat","r8024.tf.lowT.dat","r8027.tf.lowT.dat","r8031.tf.lowT.dat"]
filenamelistlow=["r8021.tf.higT.dat","r8022.tf.higT.dat","r8023.tf.higT.dat","r8024.tf.higT.dat","r8027.tf.higT.dat","r8031.tf.higT.dat"]
unstablecolorlist = [plotstyle.TS1989,plotstyle.TS1989,plotstyle.TS1989]*3
stablecolorlist   = [plotstyle.red,plotstyle.red,plotstyle.red]*3




for i in range(len(filenamelistlow)):

    filenamelow=mydir+filenamelistlow[i]
    filenamehig=mydir+filenamelisthig[i]
    label = labellist[i]
    stablecolor   = stablecolorlist[i]
    unstablecolor = unstablecolorlist[i]
    
    if not showdata[i]:
        continue
    
    df=pd.read_csv(filenamelow,sep=r"\s+",header=None,names=["x","y"],comment="#")
    
    x=df['x'].to_numpy()
    y=df['y'].to_numpy()                                                                                                                                                                   
    x=x.astype(np.double)
    y=y.astype(np.double)
    der1=np.gradient(y,x)
    der2=np.gradient(der1,x)
    cv=-x*der2

    cv = cv[2:-2]
    x  =  x[2:-2]      

    index = np.where(x>0.0001) # cutoff at very low T
    cv = cv[index]
    x  = x[index]
        
    stableindx=np.where(x<Tclist[i])
    cv_stable=cv[stableindx]
    T_stable=x[stableindx]

    unstableindx=np.where(x>=Tclist[i])
    cv_unstable=cv[unstableindx]
    T_unstable =x[unstableindx]

    maxlowTcvstable=cv_stable[-1]
    
    mplot=ax.plot(T_stable, cv_stable, label=label,marker='.',markersize=4,linestyle='-',c=stablecolor)
    current_color=mplot[0].get_color()
    mplot=ax.plot(T_unstable, cv_unstable,marker='.',markersize=4,linestyle='-',c=unstablecolor)

    axins.plot(T_stable, cv_stable, linestyle='-',c=stablecolor)

    df=pd.read_csv(filenamehig,sep=r"\s+",header=None,names=["x","y"],comment="#")
    
    x=df['x'].to_numpy()
    y=df['y'].to_numpy()                                                                                                                                                                   
    x=x.astype(np.double)
    y=y.astype(np.double)
    der1=np.gradient(y,x)
    der2=np.gradient(der1,x)
    cv=-x*der2

    cv = cv[2:-2]
    x  =  x[2:-2]      

    index = np.where(x>0.0001) # cutoff at very low T
    cv = cv[index]
    x  = x[index]
        
    stableindx=np.where(x>Tclist[i])
    cv_stable=cv[stableindx]
    T_stable=x[stableindx]

    minhigTcvstable=cv_stable[0]
    
    unstableindx=np.where(x<=Tclist[i])
    cv_unstable=cv[unstableindx]
    T_unstable =x[unstableindx]

    
    mplot=ax.plot(T_stable, cv_stable,marker='.',markersize=4,c=stablecolor,linestyle='-')
    current_color=mplot[0].get_color()
    mplot=ax.plot(T_unstable, cv_unstable,marker='.',markersize=4,c=unstablecolor,linestyle='-')

    linex=[Tclist[i],Tclist[i]]
    liney=[maxlowTcvstable,minhigTcvstable]
    mplot=ax.plot(linex,liney,marker='',alpha=1,c=stablecolor,linestyle='-')

#ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter3))
#ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter2))

#ax.set_yticks([0,-0.001,-0.002,-0.003])
#ax.set_yticklabels(['0','-1','-2','-3'],fontsize=16)

#ax.set_xticks([0,1/300,0.02,0.03,0.04])
#ax.set_xticklabels(['0',"1/300",'.02','.03','.04'],fontsize=16)

#ax.set_xlim(0.,0.05)
#ax.set_ylim(ys*0.0001,ys*0.000115)



ax.legend();  # Add a legend.



#ax.set_xlabel(r'$1/L$',fontsize=20)  # Add an x-label to the axes.
#ax.set_ylabel(r"Latent heat$[10^{-4}]$",fontsize=20)  # Add a y-label to the axes.

ax.grid(False)

print('',flush=True)
plt.savefig(savefigname)

plt.pause(100000)    
#plt.show()







#plt.show()



