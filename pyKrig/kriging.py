"""
Functions for kriging and related utilities.

Author: Padarn Wilson
"""

import numpy as np
import sklearn.gaussian_process
import seaborn as sns
import pandas as pd

def spherical_correlation(theta, d):
    theta = np.asarray(theta, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    d = np.sqrt(np.sum(d ** 2, axis=1))

    th = d * theta
    th = (1 - 3. / 2. * th + (th ** 3) / 2.)
    th[d > theta] = 0.
    th[d < 0] = 1.
    return th


def est_nugget(data, dist=10, x_col='x_m', y_col='y_m', flux_col='flux'):
    d = np.sqrt(data.x_m ** 2 + data.y_m ** 2)
    return data.flux[d <= dist].var() / 2


def krig_model(data, nugget_dist=10, x_col='x_m', y_col='y_m', flux_col='flux'):
    gp = sklearn.gaussian_process.GaussianProcess(
        thetaL=5e-3, thetaU=500,
        nugget=est_nugget(data, nugget_dist, x_col, y_col, flux_col),
        corr=spherical_correlation)
    gp.fit(data[[x_col, y_col]].values, data[flux_col].values)
    gp._xmin = data[x_col].min()
    gp._ymin = data[y_col].min()
    gp._xmax = data[x_col].max()
    gp._ymax = data[y_col].max()
    return gp

def krig_sample(model, point):
    est, std = model.predict([point], True)
    return np.random.normal(est, std**(0.5))

def krig_contour_model(model, ax, dxy=1, levels=10):
    XX, YY = np.meshgrid(
        np.arange(model._xmin, model._xmax, dxy),
        np.arange(model._ymin, model._ymax, dxy))
    XX = XX.reshape(XX.shape[0] * XX.shape[1])
    YY = YY.reshape(YY.shape[0] * YY.shape[1])
    ZZ = model.predict(np.c_[XX, YY])
    df = pd.DataFrame(np.c_[XX, YY, ZZ], columns=["x","y","flux"])
    df = df.pivot('y', 'x')
    sns.heatmap(df, ax=ax, cbar=False, cmap="coolwarm", xticklabels=False, yticklabels=False)
    ax.set_ylabel("y")
    ax.set_xlabel("x")