import matplotlib.pyplot as plt 
from matplotlib import cm 
import numpy as np
import json

def clipMatrix(array, nrange = 600):
    if nrange == -1:
        return array
    else:
        center = array.shape[-1] // 2
        return array[..., center-nrange:center+nrange]

def saveplot(path):
    plt.savefig(path,
                bbox_inches = 'tight',
                transparent = True,
                pad_inches = 0)

def convertdB(pulse, cutoff = -30):
    P = np.abs(pulse)**2
    P[P<1e-100] = 1e-100 
    P = 10*np.log10(P) 
    P[P<cutoff] = cutoff 
    
    return P
    
def plotPulse2D(fig, ax, timeArr,lengthArr,pulse, nrange = 600):
  
    ax.set_title('Pulse Evolution')
    t = clipMatrix(timeArr, nrange)*1e12
    z = lengthArr
    T, Z = np.meshgrid(t, z)
    
    P= clipMatrix(np.abs(pulse)**2, nrange)

    surf=ax.contourf(T, Z, P,levels=60)
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel('Distance [m]')
    cbar=fig.colorbar(surf, ax=ax)
    plt.show()

def plotCompare(T, Z, pulse, truthPulse, error, nrange = -1, title = None):
    T = clipMatrix(T, nrange)
    pulse = clipMatrix(pulse, nrange); truthPulse = clipMatrix(truthPulse, nrange)
    error = clipMatrix(error, nrange)
    fig, axes = plt.subplots(1,3, figsize = (18,5)) 
    
    pulse = np.abs(pulse)**2 
    truthPulse = np.abs(truthPulse)**2 
    error = np.abs(error)**2
    
    vmin = np.min(truthPulse)
    vmax = np.max(truthPulse)
    #Ground Truth 
    c1 = axes[0].contourf(T,Z,truthPulse, levels = 100, cmap = 'viridis') 
    axes[0].set_title('SSFM') 
    axes[0].set_xlabel('T') 
    axes[0].set_ylabel('Z')
    fig.colorbar(c1, ax = axes[0])
    
    #Prediction 
    c2 = axes[1].contourf(T,Z,pulse, levels = 100, cmap = 'viridis')
    axes[1].set_title('PINN') 
    axes[1].set_xlabel('T') 
    axes[1].set_ylabel('Z')
    fig.colorbar(c2, ax = axes[1])
    
    #Error 
    c3 = axes[2].contourf(T,Z,error, levels = 100, cmap = 'magma')
    axes[2].set_title('Error (MSE)') 
    axes[2].set_xlabel('T') 
    axes[2].set_ylabel('Z')
    fig.colorbar(c3, ax = axes[2])
    
    mean_error = np.mean(error)
    axes[2].text(
        0.05, 0.95, 
        f'Mean Error: {mean_error:.2e}', 
        transform=axes[2].transAxes,
        fontsize=10, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )
    plt.tight_layout()
    if title is not None:
        fig.savefig(f'{title}.png', dpi = 300, bbox_inches = 'tight')