import netCDF4

from dice.array import Array, Dimension
from dice.variable import Variable

from dice.dataset import Dataset
from dice.dataset import DatasetError

class netCDF4Array(Array):
	"""
	A netCDF4 file base array store
	"""

	def __init__(self, ncvar):

		super(netCDF4Array, self).__init__(ncvar.shape, ncvar.datatype)
		self._ncvar = ncvar

	def __setitem__(self, slices, values):		
		self._ncvar[slices] = values

	def __getitem__(self, slices):
		return self._ncvar[slices]


class netCDFVariable(Variable):
	"""
	A netCDF4 file based variable
	"""

	def __init__(self, dimensions, dtype, name=None, attributes={}, data=None):
		"""
		Just call the parent class constructor but force storage to be netCDF4Array
		"""
		
		super(Variable, self).__init__(dimensions, dtype, name=name, attributes=attributes, data=data, storage=netCDF4Array)

	@property
	def dimensions(self):
		return tuple(self._dimensions)

	@property
	def attributes(self):
		return self._attributes

	def __setitem__(self, slices, values):
		self._data[slices] = values

	def __getitem__(self, slices):
		return self._data[slices].copy()




class netCDF4Dataset(Dataset):
	"""
	A netCDF dataset implementation using the netCDF4 python module
	"""

	def __init__(self, filename, dataset=None, mode='r'):
		"""
		>>> ds = netCDF4Dataset('dice/testing/test.nc', Dataset(variables={'test1':Variable((('x', 5),('y',3)), float, attributes={'name':'test'})}, attributes={'test':'true'}))
		>>> print(ds.dimensions)
		[<Dimension: x (5) >, <Dimension: y (3) >]
		>>> print(ds.attributes)
		{u'test': u'true'}
		>>> print(ds.variables)
		{'test1': <Variable: test1 [<Dimension: x (5) >, <Dimension: y (3) >]>}
		>>> ds.variables['test1'][:] = 42.0
		>>> print(ds.variables['test1'][0,0])
		42.0
		>>> ds.close()
		>>> ds = netCDF4Dataset('dice/testing/test.nc')
		>>> print(ds.attributes)
		{u'test': u'true'}
		>>> print(ds.variables['test1'][0,0])
		42.0
		"""

		self._dimensions = []
		self._attributes = {}
		self._variables = {}


		# If there is no Dataset instance provided then assume we are opening an existing dataset
		if not dataset:
			try:
				self._ds = netCDF4.Dataset(filename, mode=mode)
			except:
				raise DatasetError("Can't open dataset {}".format(filename))

			for name, dim in self._ds.dimensions.items():
				self._dimensions.append(Dimension(name, dim.size))

			for name in self._ds.ncattrs():
				self._attributes[name] = self._ds.getncattr(name)

			for varname, var in self._ds.variables.items():

				dims = []
				for name in var.dimensions:
					for dim in self._dimensions:
						if dim.name == name:
							dims.append(dim)


				attrs = dict([(name, var.getncattr(name)) for name in var.ncattrs()])
				self._variables[varname] = Variable(dims, var.datatype, varname, attrs, data=netCDF4Array(var), dataset=self)

		# Otherwise we are creating a new dataset based on the Dataset instance
		else:
			try:
				self._ds = netCDF4.Dataset(filename, mode='w')
			except:
				raise DatasetError("Can't open dataset {} for writing".format(filename))

			# Create the dimensions
			for d in dataset.dimensions:
				self._ds.createDimension(d.name, d.size)
				self._dimensions.append(Dimension(d.name, d.size))

			# Create the global attributes
			for key, value in dataset.attributes.items():
				self._ds.setncattr(key, value)
				self._attributes[key] = value

			# Create the variables
			for name, var in dataset.variables.items():
				dims = tuple([d.name for d in var.dimensions])
				ncvar = self._ds.createVariable(name, var.dtype, dims, fill_value=False)
				ncvar[:] = var[:].copy()
				self.variables[name] = Variable(var.dimensions, var.dtype, name=name, attributes=var.attributes, 
												data=netCDF4Array(ncvar), dataset=self)


	def makefield(self):

		fields = {}

		for name, variable in self._variables:

			fields[name] = CFField(variable, self)


	@property
	def attributes(self):
		return self._attributes

	@property
	def dimensions(self):
		return self._dimensions

	@property
	def variables(self):
		return self._variables

	def close(self):
		self._ds.close()

