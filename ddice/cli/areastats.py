import sys
import os.path

import numpy as np

from ddice.dataset import Dataset, netCDF4Dataset
from ddice.variable import Variable, Dimension
from ddice.field import CFField
from ddice.field import grouping

from shapely.geometry import shape
from fiona import collection

import argparse

parser = argparse.ArgumentParser(description='Calculate area statistics')
parser.add_argument('source', help='filename:varname Source netcdf data filename and variable name to process')
parser.add_argument('-g', '--geometry', type=str,
                    help='shapefile:property The shapefile and the schema property name to use as the feature index variable')
parser.add_argument('-o', '--output', type=str, required=True,
                    help='Output filename')
args = parser.parse_args()

#print(args)

# Reading source data
filename, varname = args.source.split(':')

ds = netCDF4Dataset(filename)

field = CFField(ds.variables[varname])
print('Reading {} from {}'.format(filename, varname))

if args.geometry:
	target, keyname = args.geometry.split(':')
	print('Using geometry from {} and indexing with {}'.format(target, keyname))
else:
	target = None
	keyname = None

print('Creating geometry groups... ')
groupby = field.groupby('geometry', grouping.geometry, target=target, keyname=keyname)
print('done')

#print([(g[1].slices, g[1].weights.shape) for g in groupby.groups.items()])

print('Applying groups to field... ')
outds, outfield = field.apply(groupby, 'total')
print('done')

if target and keyname:
	outds.attributes['features_src'] = str(target)
	outds.attributes['features_key'] = str(keyname)

print('Writing to {}...'.format(args.output))
ncout = netCDF4Dataset(uri=args.output, dataset=outds)
print('done')


