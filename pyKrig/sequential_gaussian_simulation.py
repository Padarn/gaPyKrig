"""
Functions for sequential gaussian simulation and related utilities.

Author: Padarn Wilson
"""

import kriging
import numpy as np
import pandas as pd
import scipy.stats as scpstats
import seaborn as sns


def makePathAndGrid(data, xsteps, ysteps, x_col='x_m', y_col='y_m'):
    x = data[x_col]
    y = data[y_col]

    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)

    # make grid
    dx = (xmax - xmin)/xsteps
    dy = (ymax - ymin)/ysteps
    XXraw, YYraw = np.meshgrid(np.arange(xmin, xmax, dx) + dx/2, np.arange(ymin, ymax, dy) + dy/2)

    x = ((x - xmin)/(xmax-xmin) * xsteps).astype(int)
    y = ((y - ymin)/(ymax-ymin) * ysteps).astype(int)

    xx = np.arange(0, xsteps)
    yy = np.arange(0, ysteps)

    M = np.zeros((len(xx),len(yy)))

    N = len(xx) * len(yy)
    idx = np.arange(N)
    np.random.shuffle(idx)
    XX, YY = np.meshgrid(xx, yy)

    XX = XX.reshape(N)
    YY = YY.reshape(N)
    XXraw = XXraw.reshape(N)
    YYraw = YYraw.reshape(N)

    return idx, np.c_[XXraw, YYraw].T, np.c_[XX, YY].T, M

def invboxcox(y,ld):
   if ld == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(ld*y+1)/ld))

def sgs(data, xsteps=10, ysteps=10,
        nugget_dist=10, x_col='x_m', y_col='y_m', flux_col='flux',
        transform_data=True, invert_transform=True):
    x = data.x_m.values
    y = data.y_m.values
    flux = data.flux.values
    if transform_data:
        flux, L =  scpstats.boxcox(flux)
    data = pd.DataFrame(np.c_[x, y, flux], columns=[x_col, y_col, flux_col])
    new_x = []
    new_y = []
    new_flux = []
    # create array for the output
    idx, grid, indexGrid, M = makePathAndGrid(data, xsteps, ysteps)
    for step in idx :
        point = [grid[0][step], grid[1][step]]
        model = kriging.krig_model(data, nugget_dist, x_col, y_col, flux_col)
        est = kriging.krig_sample(model, point)
        indexPoint = [indexGrid[0][step], indexGrid[1][step]]
        M[indexPoint[0], indexPoint[1]] = est
        x = np.r_[x, point[0]]
        new_x.append(x[-1])
        y = np.r_[y, point[1]]
        new_y.append(y[-1])
        flux = np.r_[flux, est]
        new_flux.append(flux[-1])
        data = pd.DataFrame(np.c_[x, y, flux], columns=[x_col, y_col, flux_col])

    if invert_transform and transform_data:
        M = invboxcox(M, L)
        new_flux = invboxcox(np.array(new_flux), L)

    return grid[0,:].reshape(M.shape), grid[1,:].reshape(M.shape), M, new_x, new_y, new_flux

def show_sgs_result(r, ax):
    N = r[0].shape[0] * r[0].shape[1]
    df = pd.DataFrame(np.c_[r[0].reshape(N), r[1].reshape(N), r[2].reshape(N)], columns=['x', 'y', 'flux'])
    df = df.pivot('x', 'y')
    sns.heatmap(df, ax=ax, robust=True, cbar=False, cmap="coolwarm", xticklabels=False, yticklabels=False)
    ax.set_ylabel("y")
    ax.set_xlabel("x")
