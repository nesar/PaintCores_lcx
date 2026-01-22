import numpy as np
import matplotlib.pylab as plt
from matplotlib import ticker
np.set_printoptions(precision=3)


def set_pub():
    """ Pretty plotting changes in rc for publications
    Might be slower due to usetex=True
    
    
    plt.minorticks_on()  - activate  for minor ticks in each plot

    """
    # plt.rc('font', weight='bold')    # bold fonts are easier to see
    plt.rc('font',family='DejaVu Serif')
    # plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=False)   # Slower
    plt.rc('font',size=18)    # 18 usually
    # plt.rcParams['image.cmap'] = 'nipy_spectral_r'
    plt.rcParams['image.cmap'] = 'plasma'
    
    plt.rcParams["figure.figsize"] = [9, 5]
    plt.rcParams['axes.titlepad'] = 10
    
    
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['lines.linewidth'] = 1.5
    # lines.markersize : 10
    # xtick.labelsize : 16
    # ytick.labelsize : 16


    plt.rc('lines', lw=2, color='k', markeredgewidth=1.5) # thicker black lines
    #plt.rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines
    plt.rc('savefig', dpi=300)       # higher res outputs
    
    plt.rc('xtick', labelsize='x-large')
    plt.rc('ytick', labelsize='x-large')
    # plt.rc('axes',labelsize= 27)
    
    plt.rcParams['xtick.major.size'] = 12
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['xtick.minor.size'] = 8
    plt.rcParams['xtick.minor.width'] = 1
    
    plt.rcParams['ytick.major.size'] = 12
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['ytick.minor.size'] = 8
    plt.rcParams['ytick.minor.width'] = 1
    
    #plt.rcParams.update({'font.size': 15})
    #plt.rcParams['axes.color_cycle'] = [ 'navy', 'forestgreen', 'darkred']


#import SetPub
#SetPub.set_pub()
