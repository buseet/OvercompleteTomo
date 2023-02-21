'''
This code is a plotting reference for the examples that are shown in the paper: 
"Overcomplete Tomography: A novel approach to imaging" by B. Turunctur, A. P. Valentine, and M. Sambridge, 
which is going to be published in RAS Techniques and Instruments in 2023.

Buse Turunctur
Research School of Earth Sciences
The Australian National University
buse.turunctur@anu.edu.au
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import ConnectionPatch
import overcomplete_tomo 

class MidpointNormalize(colors.Normalize):
    """
    class to help renormalize the color scale
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
def rays_paths(ax,pthlst,txt=None):
    xmin = pthlst.min()
    xmax = pthlst.max()
    for k in range(np.shape(pthlst)[0]):
          x_values = pthlst[k][0][0], pthlst[k][1][0]
          y_values = pthlst[k][0][1], pthlst[k][1][1]
          ax.plot(x_values,y_values, 'k', alpha=0.4)
          # plt.scatter(x_values,y_values,marker='^',color='royalblue',alpha=0.8) # plot the sources and receivers
    ax.set_xlim([xmin-0.5,xmax-0.5])
    ax.set_ylim([xmin-0.5,xmax-0.5])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-16,7,txt)
    ax.invert_yaxis()
    return ax

def one_model(ax,result,colormap,colormin=None,colormax=None,txt=None):
    im = ax.imshow(result,cmap=colormap,norm=MidpointNormalize(midpoint=0))
    im.set_clim(colormin,colormax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-16,7,txt)
    return ax, im


def overcomplete_model(gs,result1,result2,colormap,colormin=None,colormax=None,txt=None):
    # Plot the overcomplete model
    ax1 = plt.subplot(gs[:, :-1])
    im = ax1.imshow((result1+result2),cmap=colormap,norm=MidpointNormalize(midpoint=0))
    ax1.set_xticks([])
    ax1.set_yticks([])
    im.set_clim(colormin,colormax)
    ax1.text(0,8,'(i)',fontsize=25)
    ax = plt.subplot(gs[:-1, -1])
    im = ax.imshow(result1,cmap=colormap,norm=MidpointNormalize(midpoint=0))
    ax.set_xticks([])
    ax.set_yticks([])
    im.set_clim(colormin,colormax)
    ax.text(0,18,'(ii)',fontsize=25)
    ax = plt.subplot(gs[-1, -1])
    im = ax.imshow(result2,cmap=colormap,norm=MidpointNormalize(midpoint=0))
    ax.set_xticks([])
    ax.set_yticks([])
    im.set_clim(colormin,colormax)
    ax1.text(-16,7, txt)
    ax.text(0,18,'(iii)',fontsize=25)
    return ax1, im