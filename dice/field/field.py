
import re

from variable.variable import Variable
from dataset.netcdf4 import netCDF4Dataset

import netCDF4


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

	def __init__(self, variable, dataset):
		"""A field encapsulates a Variable with meta-data about associated coordinate variables either
		explicitly defined or implicitely defined through conventions such as CF conventions

		>>> ds = netCDF4Dataset('testing/Rainf_WFDEI_GPCC_monthly_total_1979-2009_africa.nc')
		>>> variable = ds.variables['rainf']
		>>> print(variable)
		<Variable: [<Dimension: time (372) >, <Dimension: latitude (150) >, <Dimension: longitude (146) >]>
		>>> print(variable.attributes['units'])
		mm/day
		>>> print(variable[100,70,80])
		60.6826
		>>> f = Field(variable, ds)
		"""

		self.variable = variable
		self.dataset = dataset


	@property
	def latitudes(self):
		return None

	@property
	def longitude(self):
		return None

	@property
	def times(self):
		return None

	@property
	def vertical(self):
		return None


class CFField(Field):
	
	def __init__(self, variable, dataset):
		"""A CF Field uses the CF conventions: http://cfconventions.org/ to deterine the coordinates associated
		with a variable

		>>> ds = netCDF4Dataset('testing/Rainf_WFDEI_GPCC_monthly_total_1979-2009_africa.nc')
		>>> variable = ds.variables['rainf']
		>>> print(variable)
		<Variable: [<Dimension: time (372) >, <Dimension: latitude (150) >, <Dimension: longitude (146) >]>
		>>> print(variable[100,70,80])
		60.6826
		>>> print(variable.attributes['units'])
		mm/day
		>>> f = CFField(variable, ds)
		>>> print(f.coordinate_variables)
		{'latitude': ([1], <Variable: [<Dimension: latitude (150) >]>), 'longitude': ([2], <Variable: [<Dimension: longitude (146) >]>), 'time': ([0], <Variable: [<Dimension: time (372) >]>)}
		>>> print(f.latitudes[:5])
		[-36.25 -35.75 -35.25 -34.75 -34.25]
		>>> print(f.longitudes[:5])
		[-20.25 -19.75 -19.25 -18.75 -18.25]
		>>> print(f.times[:5])
		[datetime.datetime(1979, 1, 16, 0, 0) datetime.datetime(1979, 2, 14, 12, 0)
		 datetime.datetime(1979, 3, 16, 0, 0) datetime.datetime(1979, 4, 15, 12, 0)
		 datetime.datetime(1979, 5, 16, 0, 0)]
		"""

		super(CFField, self).__init__(variable, dataset)


		self.coordinate_variables = {}
		self.coordinates = {}

		if 'coordinates' in self.variable.attributes:
			coordinates = self.variable.attributes['coordinates'].split()
		else:
			coordinates = []


		for name, var in self.dataset.variables.items():

			if not coordinates or name in coordinates:


				if 'units' in var.attributes:
					coordinate = self.units_match(var.attributes['units'])

					if coordinate and set(var.dimensions).issubset(self.variable.dimensions):
						
						mapping = []

						for dim in var.dimensions:
							mapping.append(self.variable.dimensions.index(dim))

						self.coordinate_variables[coordinate] = (mapping, var)


	def units_match(self, units):

		for expr, coordinate in cffield_unitsmap.items():
			compiled = re.compile(expr)
			if compiled.match(units):
				return coordinate

		return None


	def coordinate(self, name):

		if name in self.coordinate_variables:
			mapping, var  = self.coordinate_variables[name]
			return var
		
		else:
			return None

	@property
	def latitudes(self):
		return self.coordinate('latitude')[:]

	@property
	def longitudes(self):
		return self.coordinate('longitude')[:]

	@property
	def times(self):
		return netCDF4.num2date(self.coordinate('time')[:], self.coordinate('time').attributes['units'])








