import sys
import netCDF4
import numpy as np
from collections import OrderedDict

from datetime import datetime as dt

import fiona
import pyproj
from shapely.geometry import shape, asShape, Polygon
from shapely.validation import explain_validity

from copy import copy

class Group(object):

	def __init__(self, slices=None, weights=None, bounds=None, key_name=None):

		self.slices = [[]] if not hasattr(slices, '__getitem__') else slices
		self.weights = [] if not hasattr(weights, '__getitem__') else weights
		self.bounds = [[]] if not hasattr(bounds, '__getitem__') else bounds

		self.key_name = key_name


def generic1d(values, keyfunc):

	groups = OrderedDict()

	# Step through all values
	for index in range(0, len(values)):

		# Generate key
		key, weight = keyfunc(values[index])

		# Add to results dictionary
		if key not in groups.keys():
			groups[key] = Group()

		groups[key].slices[0].append(index)
		groups[key].weights.append(weight)

	return groups


def all(values, bounds=False):

	def keyfunc(value):
		return values[-1],

	return generic1d(values, keyfunc)


def yearmonth(values):

	def keyfunc(value):
		return dt(value.year, value.month, 15), 1

	return generic1d(values, keyfunc)


def month(values):

	def keyfunc(value):
		return dt(values[0].year, value.month,15), 1

	return generic1d(values, keyfunc)


def year(values):

	def keyfunc(value):
		return dt(value.year,6,30), 1

	return generic1d(values, keyfunc)


def julian(values, windowsize=1):

	def keyfunc(value):
		return int(value.strftime('%j')), 1

	return generic1d(values, keyfunc)



def geometry(source, target=None, keyname=None, areas=False):

	# Split out bounds and geometries array from source
	bounds, source = source

	original_shape = source.shape
  	source = source.flatten()

	try:
		collection = fiona.open(target)

	except:

		if isinstance(target, list):
			collection = target

		elif target == None:

			collection = [{
				'geometry':asShape(Polygon([(-10,-90), (-10,90), (360, 90), (360,-90),(-10,-90)])),
				'properties':{}
			}]


	intersects = OrderedDict()


  	# First gather all source geometries into each group
  	tid = 0
  	for feature in collection:

  		geom = shape(feature['geometry'])

		if keyname:
			key = feature['properties'][keyname]
		else:
			key = tid

		intersects[key] = []

  		sid = 0
  		for s in source:

  			try:
				if s.intersects(geom):

					try:
						intersection = s.intersection(geom).area/s.area
					except:
						print(geom.area, s.area)
						intersection = 0.0

					intersects[key].append((sid, intersection))
			except:
				print(explain_validity(s))
				print(explain_validity(geom))
				sys.exit(1)

			sid += 1

		tid += 1


	groups = OrderedDict()

	# Now process each group into slices and weights
	for key in intersects:

		indices = [i[0] for i in intersects[key]]
		intersections = [i[1] for i in intersects[key]]

		w = np.zeros(source.shape, dtype=np.float32)
		w[indices] = np.array(intersections)

		# Get weights back to original shape
		w = w.reshape(original_shape)

		# Find non-zero weights
		nonzero = w.nonzero()

		# Construct the slice based on min and max non zero weight indices
		s = []
		for axis in range(len(original_shape)):
			s.append(slice(nonzero[axis].min(), nonzero[axis].max()+1))

		# Mutiply weights by areas if available
		if isinstance(areas, np.ndarray):
			w *= areas

		groups[key] = Group(s, w[s]/w[s].sum(), [])


	return groups
