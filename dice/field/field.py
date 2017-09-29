import re

from dice.variable import Variable
from dice.dataset import Dataset

import numpy as np
import netCDF4

import json


cffield_unitsmap = {
	'degrees north': 'latitude',
	'degrees_north': 'latitude',
	'degree_north': 'latitude', 
	'degree_N': 'latitude',
	'degrees_N': 'latitude',
	'degreeN': 'latitude',
	'degreesN': 'latitude',

	'degrees east': 'longitude',
	'degrees_east': 'longitude',
	'degree_east': 'longitude', 
	'degree_E': 'longitude',
	'degrees_E': 'longitude',
	'degreeE': 'longitude',
	'degreesE': 'longitude',

	'bar': 'vertical',
	'millibar': 'vertical',
	'decibar': 'vertical',
	'atmosphere': 'vertical',
	'atm': 'vertical',
	'pascal': 'vertical',
	'Pa': 'vertical',
	'hPa': 'vertical',

	'meter': 'vertical',
	'metre': 'vertical',
	'kilometer': 'vertical',
	'km': 'vertical',

	'seconds since *': 'time',
	'minutes since *': 'time',
	'hours since *': 'time',
	'days since *': 'time',
}


class FieldError(Exception):

	def __init__(self, msg):
		self.msg = msg

	def __repr__(self):
		return self.msg



class Field(object):

	def __init__(self, variable):
		"""A field encapsulates a Variable with meta-data about associated coordinate variables either
		explicitly defined or implicitely defined through conventions such as CF conventions

		>>> from dice.dataset.netcdf4 import netCDF4Dataset
		>>> ds = netCDF4Dataset('dice/testing/Rainf_WFDEI_GPCC_monthly_total_1979-2009_africa.nc')
		>>> variable = ds.variables['rainf']
		>>> print(variable)
		<netCDFVariable: rainf [(u'time', 372), (u'latitude', 150), (u'longitude', 146)]>
		>>> print(variable.attributes['units'])
		mm/day
		>>> print(variable[100,70,80].ndarray())
		[[[ 60.6825943]]]
		>>> f = Field(variable)
		"""

		self.variable = variable

		self.coordinate_variables = {}
		self.ancil_variables = {}


	def __getitem__(self, slices):
		return self.variable[slices]

	def coordinate(self, name):

		if name in self.coordinate_variables:
			mapping, var  = self.coordinate_variables[name]
			return var
		
		else:
			return None

	@property
	def shape(self):
		return self.variable.shape

	@property
	def latitudes(self):

		cvs = self.coordinate_variables
		latitudes = self.coordinate('latitude')[:]

		# If latitude and longitude map to different dimensions and we aren't already 2D
		if cvs['latitude'][0][0] != cvs['longitude'][0][0] and len(latitudes.shape) == 1:
			longitudes = self.coordinate('longitude')
			return np.broadcast_to(latitudes.ndarray().T, (longitudes.shape[0], latitudes.shape[0])).T
		
		# Otherwise we just return original array
		else:
			return latitudes

	@property
	def longitudes(self):

		cvs = self.coordinate_variables
		longitudes = self.coordinate('longitude')[:]

		# If latitude and longitude map to different dimensions and we aren't already 2D
		if cvs['latitude'][0][0] != cvs['longitude'][0][0] and len(longitudes.shape) == 1:
			latitudes = self.coordinate('latitude')
			return np.broadcast_to(longitudes.ndarray().T, (latitudes.shape[0], longitudes.shape[0]))
		
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
		>>> ds = netCDF4Dataset('dice/testing/Rainf_WFDEI_GPCC_monthly_total_1979-2009_africa.nc')
		>>> variable = ds.variables['rainf']
		>>> f = CFField(variable)
		>>> s = f.map(latitude=-34, longitude=18.5, _method='nearest_neighbour')
		>>> print s
		[slice(None, None, None), 4, 77]
		>>> f[s][:5].ndarray().flatten()
		array([  28.29000473,   39.90293503,   17.97706604,  119.78000641,
		        127.48999023], dtype=float32)
		>>> ds = netCDF4Dataset('dice/testing/pr_AFR-44_ECMWF-ERAINT_evaluation_r1i1p1_SMHI-RCA4_v1_day_19800101-19801231.nc')
		>>> variable = ds.variables['pr']
		>>> f = CFField(variable)
		>>> s = f.map(latitude=-34, longitude=18.5, _method='nearest_neighbour')
		>>> print s
		[slice(None, None, None), 27, 98]
		>>> f[s][0].ndarray()
		array([[[  3.81469727e-06]]], dtype=float32)
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

		# Create the results array
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
		>>> from dice.dataset.netcdf4 import netCDF4Dataset
		>>> ds = netCDF4Dataset('dice/testing/south_africa_1960-2015.pr.nc')
		>>> variable = ds.variables['pr']
		>>> f = CFField(variable)
		>>> s = f.subset(latitude=(-30,-20), longitude=(20,30), vertical=(1500,))


		>>> ds = netCDF4Dataset('dice/testing/pr_AFR-44_ECMWF-ERAINT_evaluation_r1i1p1_SMHI-RCA4_v1_day_19800101-19801231.nc')
		>>> variable = ds.variables['pr']
		>>> f = CFField(variable)
		>>> s = f.subset(latitude=(-30,-20), longitude=(20,25), vertical=(1000,))

		"""

		print self.coordinate_variables
		print self.ancil_variables

		mappings = {}

		for name, value in kwargs.items():
			#print name

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

			mask = variable[:] < value[0]

			if len(mask) > 1 and len(value) > 1:
				mask = np.logical_or(mask, (variable[:] > value[1]))

			#print(value, variable[:], mask)

			mapping = tuple(mapping)

			if mapping in mappings:
				mappings[mapping] = np.logical_or(mappings[mapping], mask)

			else:
				mappings[mapping] = mask


		subset = [slice(0,size) for size in list(self.shape)]
		print(subset)

		for mapping, mask in mappings.items():

			nonzero = np.nonzero(~mask)

			for i in range(0,len(nonzero)):
				start, stop = nonzero[i].min(), nonzero[i].max()
				subset[mapping[i]] = slice(start, stop+1)

		print(subset)


		result = self.__class__(self.variable[subset])



	def features(self, values=None):
		"""Return a dict of features (GeoJSON structure) for this field.  For point datasets this will be a set of 
		Point features, for gridded datasets this will be a set of simple Polygon features either inferred from the
		grid point locations, or directly from the cell bounds attributes (not implemented yet)

		The data parameter can one of None, 'last', 'all'

		The timestamp parameter can be one of None, 'last', 'range'

		>>> from dice.dataset.netcdf4 import netCDF4Dataset
		>>> ds = netCDF4Dataset('dice/testing/south_africa_1960-2015.pr.nc')
		>>> variable = ds.variables['pr']
		>>> f = CFField(variable)
		>>> ds = netCDF4Dataset('dice/testing/Rainf_WFDEI_GPCC_monthly_total_1979-2009_africa.nc')
		>>> variable = ds.variables['rainf']
		>>> f = CFField(variable)
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

				coordinates = [float(self.longitudes[feature_id]), float(self.latitudes[feature_id])]

				if 'vertical' in self.coordinate_variables:
					coordinates.extend([float(self.vertical[feature_id])])

				feature = {"type":"Feature", "geometry":{"type": "Point", "coordinates":coordinates}}

				# Feature properties come form ancilary variables
				properties = {}
				for name, mapping in self.ancil_variables.items():
					properties[name] = mapping[1][feature_id]

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






class CFField(Field):
	
	def __init__(self, variable):
		"""A CF Field uses the CF conventions: http://cfconventions.org/ to deterine the coordinates associated
		with a variable

		>>> from dice.dataset.netcdf4 import netCDF4Dataset
		>>> ds = netCDF4Dataset('dice/testing/Rainf_WFDEI_GPCC_monthly_total_1979-2009_africa.nc')
		>>> variable = ds.variables['rainf']
		>>> print(variable)
		<netCDFVariable: rainf [(u'time', 372), (u'latitude', 150), (u'longitude', 146)]>
		>>> print(variable[100,70,80].ndarray())
		[[[ 60.6825943]]]
		>>> print(variable.attributes['units'])
		mm/day
		>>> f = CFField(variable)
		>>> print(f.coordinate_variables)
		{'latitude': ([1], <netCDFVariable: latitude [(u'latitude', 150)]>), 'longitude': ([2], <netCDFVariable: longitude [(u'longitude', 146)]>), 'time': ([0], <netCDFVariable: time [(u'time', 372)]>)}
		>>> print(f.latitudes[:2,:2])
		[[-36.25 -36.25]
		 [-35.75 -35.75]]
		>>> print(f.longitudes[:2,:2])
		[[-20.25 -19.75]
		 [-20.25 -19.75]]
		>>> print(f.times[:5])
		[datetime.datetime(1979, 1, 16, 0, 0) datetime.datetime(1979, 2, 14, 12, 0)
		 datetime.datetime(1979, 3, 16, 0, 0) datetime.datetime(1979, 4, 15, 12, 0)
		 datetime.datetime(1979, 5, 16, 0, 0)]
		"""

		super(CFField, self).__init__(variable)


		if 'coordinates' in self.variable.attributes:
			coordinates = self.variable.attributes['coordinates'].split()
		else:
			coordinates = []


		for name, var in self.variable.dataset.variables.items():

			# If we have units, check if they are coordinate units, if not coordinate_name will be None
			if 'units' in var.attributes:
				coordinate_name = self.units_match(var.attributes['units'])
			else:
				coordinate_name = None

			# This could be a coordinate/ancilary variable if:
			# 1) Its in the coordinates attribute list
			# 2) Its dimensions are a reduced subset of the variables dimensions
			if (name in coordinates) or (set(var.dimensions).issubset(self.variable.dimensions) and len(var.dimensions) < len(self.variable.dimensions)):

				mapping = []
				for dim in var.dimensions:
					mapping.append(self.variable.dimensions.index(dim))

				if coordinate_name:
					self.coordinate_variables[coordinate_name] = (mapping, var)
				else:
					self.ancil_variables[name] = (mapping, var)


	def units_match(self, units):

		for expr, coordinate in cffield_unitsmap.items():
			compiled = re.compile(expr)
			if compiled.match(units):
				return coordinate

		return None




