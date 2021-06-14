from textwrap import fill
import netCDF4
import numpy as np
import sys
import glob

from ddice.array import Array, tiledArray, numpyArray
from ddice.array import reslice
from ddice.variable import Dimension, Variable

from .dataset import Dataset
from .dataset import DatasetError

MAX_MEMORY = 1000*1e6

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
				
				# Have to set fill value at creation, can't just set the attribute
				if '_FillValue' in var.attributes:
					fill_value = var.attributes['_FillValue']
				else:
					fill_value = False

				ncvar = self._ds.createVariable(name, var.dtype, dims, fill_value=fill_value)

				# Write variable attributes
				for key, value in var.attributes.items():
					if key not in ['_FillValue']:
						ncvar.setncattr(key, value)

				# Actually write the data array
				if len(dims):

					# Its a bit messy but seems to be needed to make sure that masked arrays are written
					# with the correct missing value

					# If the variable has a _FillValue attribute then we use that
					if '_FillValue' in var.attributes:
						fill_value = var.attributes['_FillValue']

					# We use object type for string arrays, so can set the fill value for strings here
					elif var.dtype == object:
						fill_value = ''

					# Else we get the default fill value for this data type from the module dictionary
					else:
						try:
							fill_value = netCDF4.default_fillvals[np.dtype(var.dtype).str.strip('<').strip('>')]
						except:
							fill_value = ''

					# Writing the whole array in one step could exceed memory limits so calculate chunck size
					size = np.cumprod(np.array(var.shape))[-1] * np.dtype(var.dtype).itemsize
					chunks = int(np.floor(size/MAX_MEMORY) + 1)

					# For now just chunk the first dimension... (come back to this)
					chunk_size = int(np.floor(var.shape[0]/chunks))

					print("writing {} chunks of size {}".format(chunks, chunk_size))

					# Now actually write the data
					for chunk in range(0, chunks):
						start = chunk*chunk_size
						end = min(start + chunk_size, var.shape[0])
						print(var.shape, chunk, start, end)
						print('fill_value', name, var.dtype, fill_value)

						try:
							ncvar[start:end] = np.ma.filled(var[start:end].ndarray(), fill_value)
						except:
							print("WARNING, problem writing this chunk", sys.exc_info())
							print('var', var[start:end].ndarray())
							print('fill_value', fill_value)


				self.variables[name] = netCDFVariable(var.dimensions, var.dtype, name=name, attributes=var.attributes, dataset=self, data=netCDF4Array(ncvar))


		# If we dont' have an existing Dataset instance or a list of dimensions we must be opening an existing uri
		else:

			# Get files list
			if isinstance(uri, list):
				files = uri

			else:
				files = glob.glob(uri)
				files.sort()

			self.files = files	

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

				# Construct dimensions
				dims = []
				for name in var.dimensions:
					for dim in self._dimensions:
						if dim.name == name:
							dims.append(dim)


				# Get variable attributes
				attrs = dict([(name, var.getncattr(name)) for name in var.ncattrs()])

				# Figure out dtype
				if var.scale and var.dtype in [np.int8, np.int16]:
					dtype = np.float32
					attrs.pop('scale_factor', None)
					attrs.pop('add_offset', None)
					
				else:
					dtype = var.dtype


				index = tuple([0]*len(var.shape))
				bounds = [(0,var.shape[i]) for i in range(len(var.shape))]
				tiledata = netCDF4Array(var)

				tiles = {index: {'bounds': bounds, 'data':tiledata}}

				data = tiledArray(var.shape, dtype, tiles=tiles)

				self._variables[varname] = netCDFVariable(dims, dtype, name=varname, attributes=attrs, data=data, storage=tiledArray, dataset=self)

			# Now try open the other files
			file_number = 1
			for file in files[1:]:

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

					# Construct the dimensions list using the dataset dimensions
					dims = []
					for name in thisvar.dimensions:
						for dim in self._dimensions:
							if dim.name == name:
								dims.append(dim)


					index = [0]*len(thisvar.shape)
					index[0] = file_number
					index = tuple(index)

					var = self.variables[varname]
					tiles = var._data._tiles

					bounds = [(0,d.size) for d in dims]
					bounds[0] = (original_size, bounds[0][1])
					shape = tuple([d.size for d in dims])

					# Create a netCDF4Array instance using the netcdf4 variable and use this as the tile data
					tiledata = netCDF4Array(thisvar)

					# If this is the time dimension then we might need to adjust the time values
					# to align with the time units of the first file
					if len(dims) == 1 and dims[0].name == aggdim and thisvar.standard_name == 'time':

						if thisvar.units != var.attributes['units']:

							tiledata = numpyArray(thisvar.shape, thisvar.dtype)

							if 'calendar' in thisvar.ncattrs():
								dates = netCDF4.num2date(thisvar[:], thisvar.units, calendar=thisvar.calendar)

							if 'calendar' in var.attributes:
								tiledata[:] = netCDF4.date2num(dates, var.attributes['units'], calendar=var.attributes['calendar'])
							else:
								tiledata[:] = netCDF4.date2num(dates, var.attributes['units'])


					# Construct the tile and add it to the tiles dict
					tiles[index] = {'bounds': bounds, 'data':tiledata}

					#Create a new tiledArray using the ammended tiles dict
					data = tiledArray(shape, dtype, tiles=tiles)

					# Creat a new variable and replace the old one
					self._variables[varname] = netCDFVariable(dims, dtype, name=varname, attributes=var.attributes, data=data, storage=tiledArray, dataset=self)

				# Increment file number as this determines the tile index
				file_number += 1


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

