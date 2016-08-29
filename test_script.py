import sys
import os
sys.path.append(os.path.join(os.path.curdir, 'pyKrig'))
import sequential_gaussian_simulation as sgs
import numpy as np
import pandas as pd
import warnings
import time

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
datafile = 'data/Ginninderra 2013 soil flux surveys_for python.xlsx'
xl = pd.ExcelFile(datafile)
data = xl.parse(xl.sheet_names[2])  # read a specific sheet to DataFrame
data = data[['x_m', 'y_m', 'raw_flux_mmol/m2/d']].rename(columns={'raw_flux_mmol/m2/d':'flux'})

print "start"
start = time.time()

sgs_result = sgs.sgs(
    data,
    dx=(data.x_m.max()-data.x_m.min())/5,
    dy=(data.y_m.max()-data.y_m.min())/5,
    invert_transform=True,
    ordinary=False)

print "Universal Kriging integral: ", sgs_result.estimate_integral(1, 1)

sgs_result = sgs.sgs(
    data,
    dx=(data.x_m.max()-data.x_m.min())/5,
    dy=(data.y_m.max()-data.y_m.min())/5,
    invert_transform=True,
    ordinary=True)

print "Ordinary Kriging integral: ", sgs_result.estimate_integral(1, 1)

print "done"
end = time.time()
print "time(s): {}".format(end - start)