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
parser.add_argument('-g', '--groupby', type=str, required=True,
                    help='group by dimension and function in the form dimension:function')
parser.add_argument('-f', '--function', type=str, required=True,
                    help='apply function')
parser.add_argument('-o', '--output', type=str, required=True,
                    help='Output filename')
args = parser.parse_args()

print(args)

# Reading source data
filename, varname = args.source.split(':')

ds = netCDF4Dataset(filename)

field = CFField(ds.variables[varname])
print('Reading {} from {}'.format(filename, varname))

dimension, group_function = args.groupby.split(':')
print('Grouping on dimension {} with function {}'.format(dimension, group_function))

func = eval('grouping.{}'.format(group_function))

groupby = field.groupby(dimension, func)

print('groupby', groupby)

print(groupby.groups)
#print([(field.times[g[1].slices[0]][0], g[1].weights.shape) for g in groupby.groups.items()])

outds, outfield = field.apply(groupby, args.function)

#if target and keyname:
#	outds.attributes['features_src'] = str(target)
#	outds.attributes['features_key'] = str(keyname)

print('Writing to {}'.format(args.output))

ncout = netCDF4Dataset(uri=args.output, dataset=outds)





