import re

import netCDF4

from field import Field



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


class CFField(Field):

	def __init__(self, variable):
		"""A CF Field uses the CF conventions: http://cfconventions.org/ to deterine the coordinates associated
		with a variable

		>>> from ddice.dataset.netcdf4 import netCDF4Dataset
		>>> ds = netCDF4Dataset(uri='ddice/testing/Rainf_WFDEI_GPCC_monthly_total_1979-2009_africa.nc')
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

		>>> print(f.latitudes[:2,:2].ndarray())
		[[-36.25 -36.25]
		 [-35.75 -35.75]]

		>>> print(f.longitudes[:2,:2].ndarray())
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
					try:
						mapping.append(self.variable.dimensions.index(dim))
					except:
						pass

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


	@property
	def times(self):
		"""Convert time coordinate values to real datetime instances based on CF
		defined units and calendar attributes on the time coordinate variable
		"""

		if 'calendar' in self.coordinate('time').attributes:
			return netCDF4.num2date(self.coordinate('time').ndarray(),
				                    self.coordinate('time').attributes['units'],
				                    self.coordinate('time').attributes['calendar'])

		else:
			return netCDF4.num2date(self.coordinate('time').ndarray(),
				                    self.coordinate('time').attributes['units'])


