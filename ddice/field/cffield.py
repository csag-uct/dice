import re

import numpy as np
import netCDF4

from ddice.variable import Dimension, Variable
from ddice.dataset import Dataset
from ddice.field import Field



cffield_unitsmap = {
	'degrees north': 'latitude',
	'degrees_north': 'latitude',
	'degree_north': 'latitude',
	'degree_N': 'latitude',
	'degrees_N': 'latitude',
	'degreeN': 'latitude',
	'degreesN': 'latitude',

	'degrees south': 'latitude',
	'degrees_south': 'latitude',

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

		# Search through all variables to try and find coordinate/ancil variables for this variable
		for name, var in self.variable.dataset.variables.items():

			# If we have units, check if they are coordinate units, if not coordinate_name will be None
			if 'units' in var.attributes:
				coordinate_name = self.units_match(var.attributes['units'])

			else:
				coordinate_name = None

			# This could be a coordinate/ancilary variable if:
			# 1) Its in the coordinates attribute list
			# 2) Its dimensions are a reduced subset of the variables dimensions

			# Can't do set operations with Dimension instances because they aren't hashable
			# So we use their string representations instead
			var_dimension_strings = [repr(d) for d in var.dimensions]
			self_dimension_strings = [repr(d) for d in self.variable.dimensions]

			if (name in coordinates) or (set(var_dimension_strings).issubset(self_dimension_strings) and (len(var.dimensions) < len(self.variable.dimensions))):

				# Now that we have found a coordinate variables, construct its dimensions mapping
				mapping = []
				for dim in var.dimensions:
					try:
						mapping.append(self.variable.dimensions.index(dim))
					except:
						pass

				# If we have a coordinate_name then we have a coordinate, otherwise its ancilary
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

		if self.coordinate('time'):

			if 'calendar' in self.coordinate('time').attributes:
				return netCDF4.num2date(self.coordinate('time').ndarray(),
					                    self.coordinate('time').attributes['units'],
					                    self.coordinate('time').attributes['calendar'])

			else:
				return netCDF4.num2date(self.coordinate('time').ndarray(),
					                    self.coordinate('time').attributes['units'])

		else:
			return None

	@classmethod
	def makefields(cls, dataset):
		"""
		Class method that constructs a dictionary of field instances within a
		dataset.
		We can't have this as a Dataset method (which would seem the most logical) because the Field
		class needs the Dataset class.  If this method were in the Dataset class then the Dataset class
		would also need the Field class and we would have a circular dependency.

		Conceptually this also makes sense as a Dataset is just a container for Variable instances. The
		Field is a higher level abstraction of the relationship between a Variable instance and it's
		host Dataset.
		"""

		fields = {}

		for name, variable in dataset.variables.items():

			try:
				field = CFField(variable)
			except:
				pass

			if len(field.coordinate_variables) > 0:
				fields[name] = field

		return fields


	@classmethod
	def merge(cls, datasets):
		"""
		Merging two datasets requires being able to distinguish between coordinate variables, ancil
		variables, and data variables.  So merging is dependent on the data convention being used,
		most common will be CF conventions (used here) but could conceivably be others, or extenstions
		of the CF conventions

		Basic strategy is to take ds1 as the master and then loop through each field (data variable)
		in ds2 checking to see if a field of the same name exists in ds1.  The logic that follows is:

		* If that field name exists in ds1 then compare the fields coordinate variables
		** If no coordinate variable values have different values then we have a potential ensemble
		merge.  This requires adding an ensemble dimension to the dataset and recreating the data variable
		with the new ensemble dimension.
		** If time is the only coordinate variable with different values then we can do a time merge
		** If other coordinate variables have different values then we abort, we can't merge different
		spatial coordinate spaces at the moment though conceptuall I guess its possible (spatial tiles?)

		* If that field name doesn't exist in the ds1 then we again compare coordinate variables
		** If all coordinate variables are identical then we can just copy the variable over because
		it can then use the original coordinate variable values in ds1.  This becomes a variable merge.

		"""

		result = Dataset(dataset=datasets[0])

		master_fields = CFField.makefields(result)

		for ds in datasets[1:]:

			fields = CFField.makefields(ds)
			print(fields)

			comparison = {}

			for name, field in fields.items():
				print(name, field)

				comparison[name] = []

				for coord, mapping in field.coordinate_variables.items():

					master_mapping = master_fields[name].coordinate_variables[coord]

					if mapping[0] == master_mapping[0]:

						# For numeric coordinate variables we can numerically compare
						if mapping[1].dtype == master_mapping[1].dtype and mapping[1].dtype not in [str, unicode]:

							if np.abs(mapping[1].ndarray() - master_mapping[1].ndarray()).max() > 0.0:

								comparison[name].append((coord, mapping))


			print(comparison)

				# Identify time merge
#				if len(different) == 1 and 'time' in different:

#					dims = master_fields[name].variable.dimensions
#					shape = np.array(master_fields[name].shape)

#					othershape = np.array(field.shape)

#					time_dim = field.coordinate_variables['time'][0][0]
#					shape[time_dim] += othershape[time_dim]

#					newdim = Dimension('time', shape[time_dim])
#					dims[time_dim] = newdim

#					print(dims)

#					newvariable = Variable()


		return result


