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

	def __init__(self, slices=None, weights=None, bounds=None, key_name=None, properties={}, schema={}):

		self.slices = [[]] if not hasattr(slices, '__getitem__') else slices
		self.weights = [] if not hasattr(weights, '__getitem__') else weights
		self.bounds = [[]] if not hasattr(bounds, '__getitem__') else bounds

		self.key_name = key_name
		self.properties = properties
		self.schema = schema


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
		return dt(value.year,12,31), 1

	return generic1d(values, keyfunc)


def julian(values, windowsize=1):

	def keyfunc(value):
		return int(value.strftime('%j')), 1

	return generic1d(values, keyfunc)


def day(values):

	def keyfunc(value):
		return dt(value.year, value.month, value.day), 1

	return generic1d(values, keyfunc)



def geometry(source, target=None, keyname=None, areas=False):

	# Split out bounds and geometries array from source
	bounds, source = source

	original_shape = source.shape
	source = source.flatten()

	# Try and open the shapefile
	try:
		collection = fiona.open(target)

	except:

		# If we fail then check if target is already a list
		if isinstance(target, list):
			collection = target

		# Finally resort to a global geometry target
		elif target == None:

			collection = [{
				'geometry':asShape(Polygon([(-10,-90), (-10,90), (360, 90), (360,-90),(-10,-90)])),
				'properties':{}
			}]



	schema = collection.schema['properties']
	properties = []

	intersects = OrderedDict()

	# First gather all source geometries into each group
	tid = 0
	for feature in collection:

		geom = shape(feature['geometry'])

		properties.append(feature['properties'])

		if keyname and keyname in feature['properties']:
			key = feature['properties'][keyname]
		else:
			key = tid

		print(key)

		# Initialise intersects for this target feature
		intersects[key] = []

		# Now we loop through all the source geomoetries
		sid = 0
		for s in source:

			try:
				if s.intersects(geom):

					# Calculate the intersection fraction
					try:
						intersection = s.intersection(geom).area/s.area
					except:
						print("WARNING: error doing intersection, setting to zero..")
						print(geom.area, s.area)

						intersection = 0.0

					# Append the source id and intersection fraction
					intersects[key].append((sid, intersection))

			# Try and display some useful diags if intersects or intersection fails
			# this usually relates to bad geometries
			except:
				print(explain_validity(s))
				print(explain_validity(geom))
				sys.exit(1)

			sid += 1

		tid += 1


	# Now we are going to actually construct the groups to return
	groups = OrderedDict()

	# Now process each group into slices and weights
	for key in intersects.keys():

		# Extract the source feature id, intersection fractions
		indices = [i[0] for i in intersects[key]]
		intersections = [i[1] for i in intersects[key]]

		# Construct the weights array
		w = np.zeros(source.shape, dtype=np.float32)
		w[indices] = np.array(intersections)

		# Get weights back to original shape
		w = w.reshape(original_shape)

		# Find non-zero weights
		nonzero = w.nonzero()

		# Only include groups where we have intersection weights
		if len(nonzero[0]):

			# Construct the slice based on min and max non zero weight indices
			s = []
			for axis in range(len(original_shape)):

				try:
					s.append(slice(nonzero[axis].min(), nonzero[axis].max()+1))
				except:
					print(len(nonzero), nonzero)

			s = tuple(s)

			# Mutiply weights by areas if available
			if isinstance(areas, np.ndarray):
				w *= areas

			groups[key] = Group(s, w[s]/w[s].sum(), [], key,
				properties=properties[list(intersects.keys()).index(key)],
				schema=schema)


	return groups
