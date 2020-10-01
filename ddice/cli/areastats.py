import sys
import os.path

import numpy as np

from ddice.dataset import Dataset, netCDF4Dataset
from ddice.variable import Variable, Dimension
from ddice.field import CFField
from ddice.field import grouping

from shapely.geometry import shape
from fiona import collection


# Reading source data
filename, varname = sys.argv[1].split(':')
print('opening', filename, varname)

ds = netCDF4Dataset(filename)
print(ds)

field = CFField(ds.variables[varname])
print('field', field.shape)


shp_filename, shp_property = sys.argv[2].split(':')
c = collection(shp_filename)
target = [shape(feature['geometry']) for feature in c]

groupby = field.groupby('geometry', grouping.geometry, target=shp_filename, keyname=shp_property)

#print([(g[1].slices, g[1].weights.shape) for g in groupby.groups.items()])

outds, outfield = field.apply(groupby, 'total')

ncout = netCDF4Dataset(uri=sys.argv[3], dataset=outds)





