"""
Functions for sequential gaussian simulation and related utilities.

Author: Padarn Wilson
"""

import kriging
import numpy as np
import pandas as pd
import scipy.stats as scpstats
import sklearn.gaussian_process

class BoxCoxInverseGP:

    def __init__(self, model, boxcoxinverse):
        self.model = model
        self.L = boxcoxinverse
        self._xmin = model._xmin
        self._ymin = model._ymin
        self._xmax = model._xmax
        self._ymax = model._ymax

    def predict(self, X):
        return invboxcox(self.model.predict(X), self.L)

class SGSResult:

    def __init__(self, model, new_x, new_y, new_flux, dx, dy, boxcoxinverse=None):
        self.new_x = new_x
        self.new_y = new_y
        self.new_flux = new_flux
        self.dx = dx
        self.dy = dy
        if boxcoxinverse:
            self.model = BoxCoxInverseGP(model, boxcoxinverse)
        else:
            self.model = model

    def estimate_integral(self, dx, dy):
        return kriging.estimate_integral(self.model, dx, dy)

    def estimate_constant_from_corners(self):
        xmin, xmax = self.model._xmin + self.dx/2, self.model._xmax - self.dx/2
        ymin, ymax = self.model._ymin + self.dy/2, self.model._ymax - self.dy/2

        return np.mean(
            self.model.predict(
                np.array([[xmin, ymin],
                 [xmin, ymax],
                 [xmax, ymin],
                 [xmax, ymax]])
            )
        )


def makePathAndGrid(data, dx, dy, x_col='x_m', y_col='y_m'):
    x = data[x_col]
    y = data[y_col]

    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)

    # make grid
    xsteps = int((xmax - xmin) / dx)
    if xsteps * dx + xmin < xmax:
        print "WARNING: dx does not exactly divide range."
    ysteps = int((ymax - ymin) / dy)
    if ysteps * dy + ymin < ymax:
        print "WARNING: dy does not exactly divide range."
    XXraw, YYraw = np.meshgrid(np.arange(xmin, xmax, dx) + dx/2, np.arange(ymin, ymax, dy) + dy/2)

    xx = np.arange(0, len(np.arange(xmin, xmax, dx) + dx/2))
    yy = np.arange(0, len(np.arange(ymin, ymax, dy) + dy/2))

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

def sgs(data, dx=10, dy=10,
        nugget_dist=10, x_col='x_m', y_col='y_m', flux_col='flux',
        transform_data=True, invert_transform=True, ordinary=False):
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
    idx, grid, indexGrid, M = makePathAndGrid(data, dx, dy)
    for step in idx :
        point = [grid[0][step], grid[1][step]]
        model = kriging.krig_model(data, nugget_dist, x_col, y_col, flux_col, ordinary=ordinary)
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

    data = pd.DataFrame(np.c_[x, y, flux], columns=[x_col, y_col, flux_col])
    model = kriging.krig_model(data, nugget_dist, x_col, y_col, flux_col, ordinary=ordinary)

    if invert_transform and transform_data:
        new_flux = invboxcox(np.array(new_flux), L)

    sgs = SGSResult(model, new_x, new_y, new_flux, dx, dy, boxcoxinverse=L)
    return sgs

