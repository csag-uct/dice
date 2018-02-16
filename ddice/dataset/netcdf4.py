import netCDF4
import numpy as np
import sys
import glob

from ddice.array import Array, tiledArray
from ddice.array import reslice
from ddice.variable import Dimension, Variable

from dataset import Dataset
from dataset import DatasetError

class netCDF4Array(Array):
	"""
	A netCDF4 file base array store
	"""

	def __init__(self, ncvar, dtype=None):

		super(netCDF4Array, self).__init__(ncvar.shape, ncvar.dtype)
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

	def __init__(self, dimensions, dtype, name=None, attributes={}, dataset=None, data=None, storage=netCDF4Array):
		"""
		Just call the parent class constructor but force storage to be netCDF4Array
		"""
		if data != None and storage == None:
			storage = data._storage

		super(netCDFVariable, self).__init__(dimensions, dtype, name=name, attributes=attributes, dataset=dataset, data=data, storage=storage)

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

	def __init__(self, uri=None, dataset=None, dimensions=(), attributes={}, variables={}, aggdim='time'):
		"""
		>>> ds = netCDF4Dataset(uri='ddice/testing/test.nc', dataset=Dataset(variables={'test1':Variable((('x', 5),('y',3)), float, attributes={'name':'test'})}, attributes={'test':'true'}))
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

		>>> ds = netCDF4Dataset(uri='ddice/testing/test.nc')
		>>> print(ds.attributes)
		{u'test': u'true'}
		>>> print(ds.variables['test1'][0,0].ndarray())
		[[ 42.]]
		"""

		# uri cannot be None, otherwise we have no-where to store this dataset
		if not uri:
			raise DatasetError("netCDF4Dataset must have a URI parameter pointing to a valid filename")


		self._dimensions = []
		self._attributes = {}
		self._variables = {}

		# Dataset argument supercedes dimensions, attributes and variables
		if dataset:

			dimensions = dataset.dimensions
			attributes = dataset.attributes
			variables = dataset.variables


		# At a minimum we need a list of dimensions to create a dataset
		if dimensions:
			try:
				self._ds = netCDF4.Dataset(uri, mode='w')
			except:
				raise DatasetError("Can't open dataset {} for writing".format(uri))

			# Create the dimensions
			for d in dimensions:
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

				# Write variable attributes
				for key, value in var.attributes.items():
					ncvar.setncattr(key, value)

				# Actually write the data array
				if len(dims):

					A = var.ndarray()

					# Its a bit messy but seems to be needed to make sure that masked arrays are written
					# with the correct missing value

					# If the variable has a _FillValue attribute then we use that
					if '_FillValue' in var.attributes:
						fill_value = var.attributes['_FillValue']

					# Else we get the default fill value for this data type from the module dictionary
					else:
						fill_value = netCDF4.default_fillvals[A.dtype.str.strip('<').strip('>')]

					# Now actually write the data
					ncvar[:] = np.ma.filled(A, fill_value)


				self.variables[name] = netCDFVariable(var.dimensions, var.dtype, name=name, attributes=var.attributes, dataset=self, data=netCDF4Array(ncvar))


		# If we dont' have an existing Dataset instance or a list of dimensions we must be opening an existing uri
		else:

			# Get files list
			files = glob.glob(uri)
			files.sort()

			print(files[0])
			# Open the first one... first
			try:
				self._ds = netCDF4.Dataset(files[0])
			except:
				raise DatasetError("Can't open dataset {}, {}".format(uri, sys.exc_info()))

			for name, dim in self._ds.dimensions.items():

				if name == aggdim:
					fixed = False
				else:
					fixed = True

				self._dimensions.append(Dimension(name, len(dim), fixed=fixed))

			for name in self._ds.ncattrs():
				self._attributes[name] = self._ds.getncattr(name)

			for varname, var in self._ds.variables.items():

				dims = []
				for name in var.dimensions:
					for dim in self._dimensions:
						if dim.name == name:
							dims.append(dim)

				attrs = dict([(name, var.getncattr(name)) for name in var.ncattrs()])

				#self._variables[varname] = netCDFVariable(dims, var.datatype, name=varname, attributes=attrs, data=netCDF4Array(var), dataset=self)
				print attrs

				index = tuple([0]*len(var.shape))
				bounds = [(0,var.shape[i]) for i in range(len(var.shape))]
				tiledata = netCDF4Array(var)

				tiles = {index: {'bounds': bounds, 'data':tiledata}}

				data = tiledArray(var.shape, var.datatype, tiles=tiles)

				self._variables[varname] = netCDFVariable(dims, var.dtype, name=varname, attributes=attrs, data=data, storage=tiledArray, dataset=self)

			# Now try open the other files
			file_number = 1
			for file in files[1:]:

				print(file)

				try:
					ds = netCDF4.Dataset(file)
				except:
					continue

				# Increase size of aggdim
				for name in ds.dimensions:
					for dim in self._dimensions:
						if dim.name == name and name == aggdim:
								original_size = dim.size
								dim.size += ds.dimensions[name].size


				# Now add each variable
				for varname, thisvar in ds.variables.items():

					# If this variable doesn't use aggdim then we can ignore it
					if aggdim not in thisvar.dimensions:
						continue

					#print('__init__', varname)

					# Construct the dimensions list using the dataset dimensions
					dims = []
					for name in thisvar.dimensions:
						for dim in self._dimensions:
							if dim.name == name:
								dims.append(dim)

					index = [0]*len(thisvar.shape)
					index[0] = file_number
					index = tuple(index)
					#print('index = ', index)

					var = self.variables[varname]
					tiles = var._data._tiles

					bounds = [(0,d.size) for d in dims]
					bounds[0] = (original_size, bounds[0][1])
					shape = tuple([d.size for d in dims])

					#print('shape', var.shape, thisvar.shape, shape)

					tiledata = netCDF4Array(thisvar)

					tiles[index] = {'bounds': bounds, 'data':tiledata}

					#print(index, tiles[index])

					data = tiledArray(shape, var.dtype, tiles=tiles)

					self._variables[varname] = netCDFVariable(dims, thisvar.dtype, name=varname, attributes=var.attributes, data=data, storage=tiledArray, dataset=self)

					#print self.variables[varname]


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

