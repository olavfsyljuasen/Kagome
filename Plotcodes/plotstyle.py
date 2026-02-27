import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler

sns.set_theme(style="whitegrid")

taylorswift = (22./255, 119./255, 87./255)
fearless    = (226./255, 156./255, 72./255)
speaknow    = (117./255, 58./255, 127./255)
red         = (166./255, 32./255, 69./255)
TS1989      = (46./255, 153./255, 249./255)
reputation  = (37./255, 38./255, 39./255)
lover       = (214./255, 54./255, 141./255)
folklore    = (171./255, 171./255, 171./255)
evermore    = (163./255, 91./255, 57./255)
midnights   = (48./255, 54./255, 117./255)
TTPD        = (239./255, 235./255, 232./255)
showgirl    = (254./255, 90./255, 0./255)

showgirl_pink = (239./255, 37./255, 121./255)
showgirl_aqua = (10./255, 198./255, 158./255)


showgirl        = (254./255, 88./255, 0./255)
showgirl_d      = (173./255, 62./255, 26./255)
showgirl_pink   = (254./255, 0./255, 124./255)
showgirl_aqua_l = (117./255, 252./255, 214./255)
showgirl_aqua_l = (142./255, 249./255, 215./255)
showgirl_aqua   = (10./255, 198./255, 158./255)
showgirl_aqua_d = (3./255, 116./255, 98./255)
showgirl_opalite = (172./255, 202./255, 215./255)

colorBlack    = ( 20./255,  20./255,  20./255)
colorDeepBlue = (  1./255, 115./255, 178./255)
colorOrange   = (222./255, 143./255,   5./255)
colorTeal     = (  2./255, 158./255, 115./255)
colorPurple   = (204./255, 120./255, 188./255)
colorBrown    = (182./255, 125./155,  77./255)



sortedColors=[ colorDeepBlue, colorOrange, colorTeal, colorPurple, colorBrown ]

TSpalette = [taylorswift, fearless, speaknow, red, TS1989, reputation, lover, folklore, evermore, midnights, TTPD]
sortedTSpalette = [TTPD, folklore, reputation, midnights, speaknow, lover, red, evermore, fearless, taylorswift, TS1989]
sortedTSpalette = [reputation, midnights, speaknow, lover, red, evermore, fearless, taylorswift, TS1989]
sortedSHOWGIRL = [showgirl, showgirl_pink, showgirl_aqua]
mpl.rcParams['axes.prop_cycle'] = cycler(color=sortedTSpalette)
#mpl.rcParams['lines.color'] = cycler(color=sortedTSpalette)

#mpl.rcParams['axes.prop_cycle'] = cycler(color=sortedColors)

R = 25
NUM_VALS = 2*R+3

replov = LinearSegmentedColormap.from_list('replov', [reputation, lover, (1, 1, 1)], N=NUM_VALS)
mpl.colormaps.register(cmap=replov)
lovrep = LinearSegmentedColormap.from_list('lovrep', [(1, 1, 1), lover, reputation], N=NUM_VALS)
mpl.colormaps.register(cmap=lovrep)
rep1989 = LinearSegmentedColormap.from_list('rep1989', [reputation, TS1989, (1, 1, 1)], N=NUM_VALS)
mpl.colormaps.register(cmap=rep1989)
TS1989rep = LinearSegmentedColormap.from_list('TS1989rep', [(1, 1, 1), TS1989, reputation], N=NUM_VALS)
mpl.colormaps.register(cmap=TS1989rep)

lover1989 = LinearSegmentedColormap.from_list('lover1989', [lover, speaknow, TS1989], N=NUM_VALS)
mpl.colormaps.register(cmap=lover1989)
speaknowtaylorswift = LinearSegmentedColormap.from_list('speaknowtaylorswift', [speaknow, TS1989, taylorswift], N=NUM_VALS)
mpl.colormaps.register(cmap=speaknowtaylorswift)
TS1989fearless = LinearSegmentedColormap.from_list('TS1989fearless', [TS1989, taylorswift, fearless], N=NUM_VALS)
mpl.colormaps.register(cmap=TS1989fearless)

fearless1989 = LinearSegmentedColormap.from_list('fearless1989', [fearless, taylorswift, TS1989], N=NUM_VALS)
mpl.colormaps.register(cmap=fearless1989)
taylorswiftspeaknow = LinearSegmentedColormap.from_list('taylorswiftspeaknow', [taylorswift, TS1989, speaknow], N=NUM_VALS)
mpl.colormaps.register(cmap=taylorswiftspeaknow)
TS1989lover = LinearSegmentedColormap.from_list('TS1989lover', [TS1989, speaknow, lover], N=NUM_VALS)
mpl.colormaps.register(cmap=TS1989lover)

showgirlmap = LinearSegmentedColormap.from_list('showgirlmap', [(1,1,1), showgirl_aqua, showgirl, showgirl_pink], N=NUM_VALS)
showgirlmap = LinearSegmentedColormap.from_list('showgirlmap', [(1,1,1), showgirl_pink, (0,0,0)], N=NUM_VALS)
showgirlmap = LinearSegmentedColormap.from_list('showgirlmap', [(0,0,0), showgirl, (1,1,1)], N=NUM_VALS)
showgirlmap = LinearSegmentedColormap.from_list('showgirlmap', [showgirl_aqua,showgirl,(1,1,1)], N=NUM_VALS)
showgirlmap = LinearSegmentedColormap.from_list('showgirlmap', [showgirl_aqua_l,showgirl,(1,1,1)], N=256)
mpl.colormaps.register(cmap=showgirlmap)
showgirlmapinv = LinearSegmentedColormap.from_list('showgirlmapinv', [(1,1,1),showgirl,showgirl_d], N=256)
mpl.colormaps.register(cmap=showgirlmapinv)

showgirlmapaqua = LinearSegmentedColormap.from_list('showgirlmapaqua', [(1,1,1),showgirl_aqua,showgirl_aqua_d], N=256)
mpl.colormaps.register(cmap=showgirlmapaqua)


spnowgrad = LinearSegmentedColormap.from_list('spnowgrad', [(1,1,1), speaknow, (0,0,0)], N=NUM_VALS)
mpl.colormaps.register(cmap=spnowgrad)

taylorswiftgrad = LinearSegmentedColormap.from_list('taylorswiftgrad', [(1,1,1), taylorswift, (0,0,0)], N=NUM_VALS)
mpl.colormaps.register(cmap=taylorswiftgrad)

bluegreengrad = LinearSegmentedColormap.from_list('bluegreengrad', [(1,1,1), TS1989,taylorswift, (0,0,0)], N=NUM_VALS)
mpl.colormaps.register(cmap=bluegreengrad)



R = 12

YNCD = LinearSegmentedColormap.from_list('YNCD', [lover, speaknow, TS1989, taylorswift, fearless], N=4*R+5)
mpl.colormaps.register(cmap=YNCD)
Daylight = LinearSegmentedColormap.from_list('Daylight', [lover, speaknow, TS1989, taylorswift, fearless, lover], N=5*R+6)
mpl.colormaps.register(cmap=Daylight)
IWCD = LinearSegmentedColormap.from_list('IWCD', [lover, speaknow, TS1989, taylorswift], N=3*R+4)
mpl.colormaps.register(cmap=IWCD)
jesuiscalm = LinearSegmentedColormap.from_list('jesuiscalm', [lover, TS1989, (1, 1, 1)], N=2*R+3)
mpl.colormaps.register(cmap=jesuiscalm)
Delicate = LinearSegmentedColormap.from_list('Delicate', [lover, TS1989, fearless], N=2*R+3)
mpl.colormaps.register(cmap=Delicate)
Lover = LinearSegmentedColormap.from_list('Lover', [(0,0,0), lover, (1,1,1)], N=2*R+3)
mpl.colormaps.register(cmap=Lover)
MidnightsLover = LinearSegmentedColormap.from_list('MidnightsLover', [(0,0,0), midnights, lover, (1,1,1)], N=2*R+3)
mpl.colormaps.register(cmap=MidnightsLover)

wlover = LinearSegmentedColormap.from_list('wlover', [(1,1,1), midnights, (0,0,0)], N=NUM_VALS)
mpl.colormaps.register(cmap=wlover)

LoverMidnights = LinearSegmentedColormap.from_list('LoverMidnights', [(0,0,0), lover, (1,1,1)], N=2*R+3)
mpl.colormaps.register(cmap=LoverMidnights)




mpl.rc('image', cmap='showgirlmap')


mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

mpl.rcParams.update({'font.size': 16})

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.left'] = True


if __name__ == '__main__':
    plt.figure()
    plt.plot(np.linspace(0, 10, 10), 'o-', label = 'Taylor Swift')
    plt.plot(2*np.linspace(0, 10, 10), 'o-', label = 'Fearless')
    plt.plot(3*np.linspace(0, 10, 10), 'o-', label = 'Speak Now')
    plt.plot(4*np.linspace(0, 10, 10), 'o-', label = 'Red')
    plt.plot(5*np.linspace(0, 10, 10), 'o-', label = '1989')
    plt.plot(6*np.linspace(0, 10, 10), 'o-', label = 'reputation')
    plt.plot(7*np.linspace(0, 10, 10), 'o-', label = 'Lover')
    plt.legend()

    plt.figure()
    plt.plot(np.linspace(0, 10, 10), 'o-', color=taylorswift, label = 'Taylor Swift')
    plt.plot(2*np.linspace(0, 10, 10), 'o-', color=fearless, label = 'Fearless')
    plt.plot(3*np.linspace(0, 10, 10), 'o-', color=speaknow, label = 'Speak Now')
    plt.plot(4*np.linspace(0, 10, 10), 'o-', color=red, label = 'Red')
    plt.plot(5*np.linspace(0, 10, 10), 'o-', color=TS1989, label = '1989')
    plt.plot(6*np.linspace(0, 10, 10), 'o-', color=reputation, label = 'reputation')
    plt.plot(7*np.linspace(0, 10, 10), 'o-', color=lover, label = 'Lover')
    plt.legend()
    plt.show()


    def f(x, y):
        return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

    x = np.linspace(0, 5, 50)
    y = np.linspace(0, 5, 40)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    plt.contourf(X, Y, Z, 20)
    plt.colorbar()
    plt.show()
