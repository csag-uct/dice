import re
import copy

from dice.array import numpyArray
from dice.variable import Variable, Dimension
from dice.dataset import Dataset

import grouping

import numpy as np
import netCDF4

import json




class FieldError(Exception):
	"""
	Pretty generic Exception subclass for Field
	"""

	def __init__(self, msg):
		self.msg = msg

	def __repr__(self):
		return self.msg


class GroupBy(object):

	def __init__(self, coordinate, groups):

		self.coordinate = coordinate
		self.groups = groups



class Field(object):

	def __init__(self, variable):
		"""A field encapsulates a Variable with meta-data about associated coordinate variables either
		explicitly defined or implicitely defined through conventions such as CF conventions.  It also attempts
		to identify ancilary variables which are variables that associated with a data variable such as station names,
		grid altitudes, etc.  

		This class should not be used directly because it doesn't implement any conventions and so does not
		identify coordinate variables or ancilary variables.  A derived class like CFField should be used instead.

		"""

		# The variable instance the field references
		self.variable = variable


		# Dictonaries, indexed by variable name of the coordinate variables and ancilary variables
		# These dictionaries are index by name and contain tuples of dimension mappings and variable references

		self.coordinate_variables = {}

		self.ancil_variables = {}



	# Return an index based subset (view) of the field
	def __getitem__(self, slices):

		"""This just delegates to the underlying variable implementation of __getitem__"""

		return self.variable[slices]



	# Tries to find a coordinate with provided name and returns associated variable or None
	def coordinate(self, name):

		if name in self.coordinate_variables:

			mapping, var  = self.coordinate_variables[name]
			return var
		
		else:
			return None


	# Shape delegates to the upstream variable shape property
	@property
	def shape(self):
		return self.variable.shape


	# Return latitude coordinate Variable instance, but try to broadcast to 2D if possible
	@property
	def latitudes(self):

		cvs = self.coordinate_variables
		latitudes = self.coordinate('latitude')

		# If latitude and longitude map to different dimensions and we aren't already 2D
		if cvs['latitude'][0][0] != cvs['longitude'][0][0] and len(latitudes.shape) == 1:
		
			longitudes = self.coordinate('longitude')
		
			latitudes2d = Variable([latitudes.dimensions[0], longitudes.dimensions[0]], latitudes.dtype)
			latitudes2d[:] = np.broadcast_to(latitudes.ndarray().T, (longitudes.shape[0], latitudes.shape[0])).T
			return latitudes2d
		

		# Otherwise we just return original array
		else:
			return latitudes


	@property
	def longitudes(self):

		cvs = self.coordinate_variables
		longitudes = self.coordinate('longitude')

		# If latitude and longitude map to different dimensions and we aren't already 2D
		if cvs['latitude'][0][0] != cvs['longitude'][0][0] and len(longitudes.shape) == 1:

			latitudes = self.coordinate('latitude')
			
			longitudes2d = Variable([latitudes.dimensions[0], longitudes.dimensions[0]], latitudes.dtype)
			longitudes2d[:] = np.broadcast_to(longitudes.ndarray().T, (latitudes.shape[0], longitudes.shape[0]))
			return longitudes2d

		# Otherwise we just return original array
		else:
			return longitudes


	@property
	def times(self):
		return netCDF4.num2date(self.coordinate('time').ndarray(), self.coordinate('time').attributes['units'])


	@property
	def vertical(self):
		return self.coordinate('vertical')


	def map(self, **kwargs):
		"""Maps from world coordinates to array indices

		>>> from dice.dataset.netcdf4 import netCDF4Dataset
		>>> from dice.field import CFField

		>>> ds = netCDF4Dataset('dice/testing/Rainf_WFDEI_GPCC_monthly_total_1979-2009_africa.nc')
		>>> variable = ds.variables['rainf']
		>>> f = CFField(variable)
		>>> s = f.map(latitude=-34, longitude=18.5, _method='nearest_neighbour')
		>>> print s
		[slice(None, None, None), 4, 77]
		>>> print f.latitudes[s[1],s[2]].ndarray()
		[[-34.25]]
		>>> print f.longitudes[s[1], s[2]].ndarray()
		[[ 18.25]]

		>>> ds = netCDF4Dataset('dice/testing/pr_AFR-44_ECMWF-ERAINT_evaluation_r1i1p1_SMHI-RCA4_v1_day_19800101-19801231.nc')
		>>> variable = ds.variables['pr']
		>>> f = CFField(variable)
		>>> s = f.map(latitude=-34, longitude=18.5, _method='nearest_neighbour')
		>>> print s
		[slice(None, None, None), 27, 98]
		>>> print f.latitudes[s[1],s[2]].ndarray()
		[[-33.88]]
		>>> print f.longitudes[s[1], s[2]].ndarray()
		[[ 18.48]]

		>>> ds = netCDF4Dataset('dice/testing/south_africa_1960-2015.pr.nc')
		>>> variable = ds.variables['pr']
		>>> f = CFField(variable)
		
		>>> s = f.map(latitude=-34, longitude=18.5, _method='nearest_neighbour')
		>>> print ds.variables['name'][:][s[1]].ndarray()
		[u'KENILWORTH RACE COURSE ARS']

		>>> s = f.map(id='0021178AW')
		>>> print ds.variables['name'][:][s[1]].ndarray()
		[u'CAPE TOWN WO']
		"""

		# Create the results array to hold the array indices (slices)
		result = [slice(None)]*len(self.variable.shape)

		# We need to keep track of processed arguments
		done = []

		# Step through all the provided arguments
		for arg, value in kwargs.items():

			# Ingore special _method argument
			if arg in ['_method']:
				continue

			# We may have processed an argument out of turn so need to check
			if arg in done:
				continue

			# We build up a list of complimentary arguments
			args = {}

			# Check if the arg is in the coordinate_variables dict
			if arg in self.coordinate_variables:
				mapping = self.coordinate_variables[arg]
				dims, coord_var = mapping
				shape = coord_var.shape
				args[arg] = mapping

				# Now search the other coordinate variables for the same mapping (and same dtype just in case!)
				for name, othermapping in self.coordinate_variables.items():
					
					# It needs to be in the arguments list, and have the same dimensions
					if name in kwargs and name != arg and othermapping[0] == dims:
						args[name] = othermapping
			
			# else check in ancilary variables
			elif arg in self.ancil_variables:
				mapping = self.ancil_variables[arg]
				dims, ancil_var = mapping
				shape = ancil_var.shape
				args[arg] = mapping
			
			# Else just quietly ignore
			else:
				continue

			# We are going to accumulate euclidian distance for numeric coordinate variables
			distance = np.zeros(shape)

			for name, mapping in args.items():
				
				# There should be a better way to differentiate between numeric and string methods
				try:
					distance += np.power(mapping[1].ndarray() - float(kwargs[name]),2)
				except:
					try:
						distance = (mapping[1].ndarray() != kwargs[name]).astype(np.int)
					except:
						pass

			# Calculate final euclidian distance
			distance = np.sqrt(distance)

			# Find the minimum and map indices back into variable indices
			i = 0
			for index in np.unravel_index(np.argmin(distance), shape):
				result[dims[i]] = index
				i += 1

			done.extend(args.keys())

		return result


	def subset(self, **kwargs):
		"""
		Subsets a field and returns a new Dataset instance hosting the subsetted field as well as 
		subsetted coordinate and ancilary variables.  Subsetting is lazy (applies slices to original variable
		arrays) so this is a cheap operation.

		>>> from dice.dataset.netcdf4 import netCDF4Dataset
		>>> from dice.field import CFField

		>>> ds = netCDF4Dataset('dice/testing/south_africa_1960-2015.pr.nc')
		>>> variable = ds.variables['pr']
		>>> f = CFField(variable)

		>>> s = f.subset(latitude=(-30,-20), longitude=(20,30), vertical=(1500,))
		>>> print(s.shape)
		(20454, 348)
		>>> print(s.longitudes.ndarray().min(), s.longitudes.ndarray().max())
		(23.554399, 29.983801)

		>>> ds = netCDF4Dataset('dice/testing/pr_AFR-44_ECMWF-ERAINT_evaluation_r1i1p1_SMHI-RCA4_v1_day_19800101-19801231.nc')
		>>> variable = ds.variables['pr']
		>>> f = CFField(variable)
		>>> s = f.subset(latitude=(-30,-20), longitude=(20,25), vertical=(1000,))
		>>> print(s.longitudes.ndarray().min(), s.longitudes.ndarray().max())
		(20.240000000000002, 24.640000000000011)
		"""

		mappings = {}

		for name, value in kwargs.items():
			#print name, value

			# First convert single values to tuples
			if type(value) not in [tuple, list]:
				value = tuple([value])

			# Check if variable is a coordinate variable
			if name in self.coordinate_variables:
				mapping, variable = self.coordinate_variables[name]

			# Check if its an ancilary variable
			elif name in self.ancil_variables:
				mapping, variable = self.ancil_variables[name]

			else:
				continue

			#print variable, mapping, value

			vals = variable.ndarray()
			mask = vals < value[0]

			if len(mask) > 1 and len(value) > 1:
				mask = np.logical_or(mask, (vals > value[1]))

			#print(value, variable[:], mask)

			mapping = tuple(mapping)

			if mapping in mappings:
				mappings[mapping] = np.logical_or(mappings[mapping], mask)

			else:
				mappings[mapping] = mask


		subset = [slice(0,size) for size in list(self.shape)]
		#print(subset)

		for mapping, mask in mappings.items():

			nonzero = np.nonzero(~mask)

			for i in range(0,len(nonzero)):
				start, stop = nonzero[i].min(), nonzero[i].max()

			 	# 1D mappings could be spare masks, rather than ranges
			 	if len(nonzero) == 1:
			 		if stop - start + 1 > nonzero[0].shape[0]:
			 			subset[mapping[i]] = nonzero[0]

				# otherwise we construct a slice
			 	else:
					subset[mapping[i]] = slice(start, stop+1)


		variable = self.variable[subset]
		variables = dict({self.variable.name: variable})

		for coord_name, map_var in self.coordinate_variables.items():
			coord_slice = []
			
			for d, s in zip(self.variable.dimensions, subset):
				if d in map_var[1].dimensions:
					coord_slice.append(s)

			variables[map_var[1].name] = map_var[1][coord_slice]


		for ancil_name, map_var in self.ancil_variables.items():
			ancil_slice = []
			
			for d, s in zip(self.variable.dimensions, subset):
				if d in map_var[1].dimensions:
					ancil_slice.append(s)

			variables[map_var[1].name] = map_var[1][ancil_slice]


		dataset = Dataset(variable.dimensions, self.variable.dataset.attributes, variables)

		result = self.__class__(variable)

		return result


	def features(self, values=None):
		"""Return a dict of features (GeoJSON structure) for this field.  For point datasets this will be a set of 
		Point features, for gridded datasets this will be a set of simple Polygon features either inferred from the
		grid point locations, or directly from the cell bounds attributes (not implemented yet)

		The data parameter can one of None, 'last', 'all'

		The timestamp parameter can be one of None, 'last', 'range'

		>>> from dice.dataset.netcdf4 import netCDF4Dataset
		>>> from dice.field import CFField

		>>> ds = netCDF4Dataset('dice/testing/south_africa_1960-2015.pr.nc')
		>>> variable = ds.variables['pr']
		>>> f = CFField(variable)
		>>> features = f.features()
		>>> print(features['features'][0]['properties'])
		{u'name': u'BOSCHRAND', u'id': u'0585409_W'}
		>>> ds = netCDF4Dataset('dice/testing/Rainf_WFDEI_GPCC_monthly_total_1979-2009_africa.nc')
		>>> variable = ds.variables['rainf']
		>>> f = CFField(variable)
		>>> features = f.features()
		"""

		# First we need to determine the type of grid we have.  If latitude and longitude are 1D and both
		# map to the same variable dimension then we have a discrete points dataset, otherwise a gridded dataset

		result = {"type":"FeatureCollection", "features":[]}


		if values == 'last':

			s = [slice(None)]*len(self.variable.shape)
			timedim = self.coordinate_variables['time'][0][0]

			s[timedim] = slice(-1,None)
			data = self.variable[s]
			last_time = self.times[-1].isoformat()

		elif values == 'last_valid':
			s = [slice(None)]*len(self.variable.shape)
			data = self.variable[s][:]


		if len(self.latitudes.shape) == 1 and len(self.longitudes.shape) == 1 and (self.coordinate_variables['latitude'][0] == self.coordinate_variables['longitude'][0]):

			feature_dim = self.coordinate_variables['latitude'][0][0]

			# Iterate through features by iterating through longitude
			for feature_id in range(self.longitudes.shape[0]):

				coordinates = [float(self.longitudes.ndarray()[feature_id]), float(self.latitudes.ndarray()[feature_id])]

				if 'vertical' in self.coordinate_variables:
					coordinates.extend([float(self.vertical.ndarray()[feature_id])])

				feature = {"type":"Feature", "geometry":{"type": "Point", "coordinates":coordinates}}

				# Feature properties come form ancilary variables
				properties = {}
				for name, mapping in self.ancil_variables.items():
					properties[name] = mapping[1][feature_id].ndarray()[0]

				# Process data based properties
				if values == 'last':
					s[feature_dim] = feature_id
					properties['value'] = (last_time, float(data[s][0]))

				elif values == 'last_valid':
					s[feature_dim] = feature_id
					subset = data[s]

					if np.ma.count(subset):
						last_time = np.ma.masked_array(self.times[:], mask=np.ma.getmaskarray(data[s])).compressed()[-1]
						properties['value'] = (last_time, float(data[s].compressed()[-1]))
					else:
						properties['value'] = (None, None)


				feature['properties'] = properties
				result['features'].append(feature)


		else:
			return {'featureType':'gridded'}

		return result



	def groupby(self, coordinate, func):

		"""
		Generate groups (subsets) across the specified coordinate using the grouping function func.  Returns a GroupBy 
		instance which captures the coordinate name and a dictionary of slices indexed by the group index.

		>>> from dice.dataset.netcdf4 import netCDF4Dataset
		>>> from dice.field import CFField
		>>> import numpy as np

		>>> ds = netCDF4Dataset('dice/testing/south_africa_1960-2015.pr.nc')
		>>> variable = ds.variables['pr']
		>>> f = CFField(variable)

		>>> groups = f.groupby('time', grouping.yearmonth)
		>>> print(groups.coordinate, len(groups.groups))
		('time', 672)

		"""

		# Find the coordinate mapping for the request coordinate, if it exists
		if coordinate in self.coordinate_variables:
			mapping, coordinate_variable = self.coordinate_variables[coordinate]

		else:
			raise FieldError("Can't find coordinate {} for groupby".format(coordinate))


		# If coordinate is time then we need to apply CF conventions to get real times... this breaks our data model a bit :-0
		if coordinate == 'time':
			coordinate_values = netCDF4.num2date(coordinate_variable[:].ndarray(), coordinate_variable.attributes['units'])

		else:
			coordinate_values = coordinate_variable[:]


		# Apply grouping function to coordinate values
		groups = func(coordinate_values)
		

		# We are going to need to construct slices on the original field
		s = [slice(None)] * len(self.shape)
		
		# Process each group
		for key, group in groups.items():

			# Try and simplify sequence to a range
			if (group[-1] - group[0] + 1) == len(group):
				groups[key] = slice(group[0], group[-1]+1)

			# Currenly this only works for single dimension mapping!
			s[mapping[0]] = groups[key]

			# Save the slices for this group
			groups[key] = copy.copy(s)


		return GroupBy(coordinate, groups)



	def apply(self, groups, func):
		"""
		Apply the func func(ndarray, axis=0) to each group and construct a new dataset and field as a result

		>>> from dice.dataset.netcdf4 import netCDF4Dataset
		>>> from dice.field import CFField
		>>> import numpy as np

		>>> ds = netCDF4Dataset('dice/testing/south_africa_1960-2015.pr.nc')
		>>> variable = ds.variables['pr']
		>>> f = CFField(variable)

		>>> ds, ff = f.apply(f.groupby('time', grouping.year), np.ma.sum)
		>>> print(ds.variables.keys())
		[u'pr', u'name', 'vertical', 'longitude', 'time', 'latitude', u'id']

		"""

		# First check if we can get the coordinate variable
		if groups.coordinate in self.coordinate_variables:
			mapping, coordinate_variable = self.coordinate_variables[groups.coordinate]

		else:
			raise FieldError("Can't find coordinate {} in field for groupby method".format(groups.coordinate))

		
		# The new variable with have the same dimensions except for the grouping coordinate which is replaced
		# by the group size

		dimensions = list(self.variable.dimensions)
		dimensions[mapping[0]] = Dimension(self.variable.dimensions[mapping[0]].name, len(groups.groups))

		# Create the variable using numpy storage for now
		variable = Variable(dimensions, self.variable.dtype, name=self.variable.name, attributes=self.variable.attributes, storage=numpyArray)
		variables = {self.variable.name: variable}

		# Now we create the coordinate variables which are just references to the existing coordinate
		# variables except for the grouping coordinate which needs to be a new variable
		for name, var in self.coordinate_variables.items():

			if name == groups.coordinate:
				variables[name] = Variable([dimensions[mapping[0]]], var[1].dtype, attributes=var[1].attributes)

			else:
				variables[name] = var[1]


		# Reference the ancilary variables in the same way
		for name, var in self.ancil_variables.items():
			variables[name] = var[1]


		# Now we actually iterate through the groups applying the function and writing results to the
		# the new variable and coordinate values to the new coordinate variable
		
		s = [slice(None)] * len(dimensions)

		i = 0
		for key, group in groups.groups.items():

			# Set the slice
			s[mapping[0]] = i

			# Apply the funcion and assign to the new variable
			variable[i] = func(self.variable[s].ndarray(), axis=mapping[0])

			# Extract the last coordinate value from the original coordinate variable for this group
			variables[groups.coordinate][i] = coordinate_variable.ndarray()[group[mapping[0]].stop - 1]
			
			i += 1


		dataset = Dataset(dimensions, self.variable.dataset.attributes, variables)

		return dataset, self.__class__(variable)












