"""
Functions for sequential gaussian simulation and related utilities.

Author: Padarn Wilson
"""

import kriging
import numpy as np
import pandas as pd
import scipy.stats as scpstats
import sklearn.gaussian_process
import seaborn as sns
import scipy.spatial as spatial

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


#Ivan added to plot data - doesn't work
#This needs to be given a model? cAN'T GET TO WORK YET
def show_sgs_result(r, ax):
    N = r[0].shape[0] * r[0].shape[1]
    df = pd.DataFrame(np.c_[r[0].reshape(N), r[1].reshape(N), r[2].reshape(N)], columns=['x', 'y', 'flux'])
    df = df.pivot('x', 'y')
    sns.heatmap(df, ax=ax, robust=True, cbar=False, cmap="coolwarm", xticklabels=False, yticklabels=False)
    ax.set_ylabel("y")
    ax.set_xlabel("x")

def in_hull(p, hull):
    if not isinstance(hull,spatial.Delaunay):
        hull = spatial.Delaunay(hull)
    
        return hull.find_simplex(p)>=0
	
def makePathAndGrid(data, dx, dy, x_col='x_m', y_col='y_m'):
    x = data[x_col]
    y = data[y_col]

    #xmin = min(x)
    #xmax = max(x)
    #ymin = min(y)
    #ymax = max(y)
    xmin = 0.0
    xmax = 120.0
    ymin = -30.0
    ymax = 30.0

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

    xy_data = np.column_stack((x,y))
    #M = np.zeros((len(xx),len(yy)))  #### see below for new M

    N = len(xx) * len(yy)
    #idx = np.arange(N)
    XX, YY = np.meshgrid(xx, yy)

    XX = XX.reshape(N)
    YY = YY.reshape(N)
    XXraw = XXraw.reshape(N)
    YYraw = YYraw.reshape(N)
	#Ivan added to restrict data to convex hull
    grid = np.c_[XXraw, YYraw].T
    gridpts = np.zeros((len(grid[0]),2))
    #indexGrid = np.c_[XX, YY].T #old indexGrid
    for i in range(0,len(grid[0])):
        gridpts[i,0] = grid[0,i]
        gridpts[i,1] = grid[1,i]
    inhull_test = in_hull(gridpts,xy_data)
    gridpts = gridpts[in_hull(gridpts,xy_data)]
    M = np.zeros((len(gridpts)))#,2))
    idx = np.arange(len(gridpts))
    np.random.shuffle(idx)
    indexGrid = np.zeros((len(gridpts)))
    indexGrid = range(0,len(gridpts)) 

    return idx, gridpts, indexGrid, M

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
    idx, gridpts, indexGrid, M = makePathAndGrid(data, dx, dy) #grid
    for step in idx :
        #point = [grid[0][step], grid[1][step]]
        point = gridpts[step]
        model = kriging.krig_model(data, nugget_dist, x_col, y_col, flux_col, ordinary=ordinary)
        est = kriging.krig_sample(model, point)
        #indexPoint = [indexGrid[step,0]] ###
        ###M is hte array of estimates
        #M[indexPoint[0], indexPoint[1]] = est
        M[indexGrid[step]] = est
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

    sgsx = SGSResult(model, new_x, new_y, new_flux, dx, dy, boxcoxinverse=L)
    return sgsx
