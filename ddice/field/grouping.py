import netCDF4
import numpy as np
from collections import OrderedDict

import fiona
import shapely

from copy import copy



def generic(values, keyfunc):

	result = OrderedDict()

	# Step through all values
	for index in range(0, len(values)):

		# Generate key
		key = keyfunc(values[index])

		# Add to results dictionary
		if key not in result.keys():
			result[key] = [[index]]
		else:
			result[key][0].append(index)

	return result, None


def all(values, bounds=False):

	def keyfunc(value):
		return 'all'

	return generic(values, keyfunc)


def yearmonth(values, bounds=False):

	def keyfunc(value):
		return value.year, value.month

	return generic(values, keyfunc)


def month(values, bounds=False):

	def keyfunc(value):
		return value.month

	return generic(values, keyfunc)


def year(values, bounds=False):

	def keyfunc(value):
		return value.year

	return generic(values, keyfunc)


def geometry(source, target=None, key_property=None):

	print(source.shape, target)

	if isinstance(target, str) or isinstance(target, unicode):
		collection = fiona.open(target)

	else:
		collection = target

	groups = OrderedDict()
	weights = []

	original_shape = source.shape
  	source = source.flatten()

  	# First gather all source geometries into each group
  	tid = 0
  	for feature in collection:

  		geom = shapely.geometry.shape(feature['geometry'])

  		sid = 0
  		for s in source:

			if s.intersects(geom):

				intersection = s.intersection(geom).area/s.area

				if key_property:
					key = feature['properties'][key_property]
				else:
					key = tid

				if key not in groups:
					groups[key] = [(sid, intersection)]
				else:
					groups[key].append((sid, intersection))

			sid += 1

		tid += 1


	# Now process each group into slices and weights
	for key in groups:

		indices = [i[0] for i in groups[key]]
		intersections = [i[1] for i in groups[key]]

		w = np.zeros(source.shape, dtype=np.float32)
		w[indices] = np.array(intersections)

		w = w.reshape(original_shape)
		nonzero = w.nonzero()

		print(key, nonzero)

		# Construct the slice based on min and max non zero weight indices
		s = []

		for axis in range(len(original_shape)):
			s.append(slice(nonzero[axis].min(), nonzero[axis].max()+1))


		# Accumulate the list of groups
		groups[key] = s

		# Subset and normalize the weights
		w = w[s]/w[s].sum()
		weights.append(w)


	print groups, [w.shape for w in weights]
	return groups, weights
