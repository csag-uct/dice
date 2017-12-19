import netCDF4

from dice.array import Array
from dice.array import reslice
from dice.variable import Dimension, Variable

from dataset import Dataset
from dataset import DatasetError

class netCDF4Array(Array):
	"""
	A netCDF4 file base array store
	"""

	def __init__(self, ncvar):

		super(netCDF4Array, self).__init__(ncvar.shape, ncvar.datatype)
		self._data = ncvar

#	def __setitem__(self, slices, values):		
#		self._ncvar[slices] = values

	def __getitem__(self, slices):

		shape, view = reslice(self.shape, self._view, slices)

		result = self.__class__(self._data)
		result._view = view
		
		return result


class netCDFVariable(Variable, object):
	"""
	A netCDF4 file based variable
	"""

	def __init__(self, dimensions, dtype, name=None, attributes={}, dataset=None, data=None):
		"""
		Just call the parent class constructor but force storage to be netCDF4Array
		"""
		
		super(netCDFVariable, self).__init__(dimensions, dtype, name=name, attributes=attributes, dataset=dataset, data=data, storage=netCDF4Array)

#	@property
#	def dimensions(self):
#		return tuple(self._dimensions)

#	@property
#	def attributes(self):
#		return self._attributes

#	def __setitem__(self, slices, values):
#		self._data[slices] = values

#	def __getitem__(self, slices):
#		return self._data[slices].copy()




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
		{'test1': <netCDFVariable: test1 [('x', 5), ('y', 3)]>}
		>>> ds.variables['test1'][:] = 42.0
		>>> print(ds.variables['test1'][0,0].ndarray())
		[[ 42.]]
		>>> ds.close()
		>>> ds = netCDF4Dataset('dice/testing/test.nc')
		>>> print(ds.attributes)
		{u'test': u'true'}
		>>> print(ds.variables['test1'][0,0].ndarray())
		[[ 42.]]
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
				self._variables[varname] = netCDFVariable(dims, var.datatype, varname, attrs, data=netCDF4Array(var), dataset=self)

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
				ncvar[:] = var.ndarray()

				# Write variable attributes
				for key, value in var.attributes.items():
					ncvar.setncattr(key, value)

				self.variables[name] = netCDFVariable(var.dimensions, var.dtype, name=name, attributes=var.attributes, dataset=self, data=netCDF4Array(ncvar))


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

