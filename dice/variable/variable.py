from abc import ABCMeta, abstractmethod

from copy import copy

from dice.array import Array
from dice.array import numpyArray


class Dimension(object):
	"""
	A dimension defines the axis of a multi-dimensional variable.  It simply has a name
	and a size.  A dimension can be fixed size (default) or not fixed size indicating whether 
	the size can or cannot be changed after creation.  This is relevant to some array storage 
	formats	that do not allow dimension sizes to change after creation
	"""

	def __init__(self, name, size, fixed=True):
		"""
		A Dimension is created with a name and size and is fixed in size by default:

		>>> d = Dimension('test', 10)
		>>> print(d)
		<Dimension: test (10) >

		Setting fixed to False lets the size be changed
		>>> d = Dimension('test2', 10, fixed=False)
		>>> d.size = 20
		>>> print(d)
		<Dimension: test2 (20) >
		"""

		self.name = name
		self._size = size

		# fixed must be boolean
		if type(fixed) == bool:
			self.fixed = fixed
		else:
			raise TypeError('unlimited must be boolean, True or False')


	# At some point we might want to return a calculated value, so writing this as a property function
	@property
	def size(self):
		return self._size


	@size.setter
	def size(self, size):
		"""
		Changes the size of a dimension. Raises an exception if the dimension is fixed.  Checks for positve
		integer values
		"""

		if self.fixed:
			raise ValueError('cannot change the size of fixed dimension {}'.format(self.name))

		# size must be a postive integer
		if size >= 0 and type(size) == int:
			self._size = size
		else:
			raise TypeError('size must be positive int')



	def __repr__(self):
		return "<Dimension: {} ({}) >".format(self.name, self.size)

	def asjson(self):
		return {'name':self.name, 'size':self.size}


class Variable(object):
	"""
	A variable encapsulates an array with named dimensions and attributes
	"""

	def __init__(self, dimensions, dtype, name=None, attributes={}, dataset=None, data=None, storage=numpyArray):
		"""
		Create a new variable instance.  The dimensions parameter must be a tuple consisting of either
		instances of Dimension or 2-tuples of the form (name, size) or (name, size, fixed)

		>>> V = Variable((('x', 5),('y',3)), float, 'V')
		>>> print(V.name)
		V
		>>> print(V)
		<Variable: V [('x', 5), ('y', 3)]>
		>>> V.shape
		(5, 3)
		>>> V[:] = 42.0
		>>> V[2,1]
		<Variable: V [('x', 1), ('y', 1)]>
		>>> V.units = 'kg/m2/s'
		"""

		self.dimensions = []
		self._dtype = dtype
		self.name = name
		self.dataset = dataset
		self._data = False

		# dimensions argument should be an iterable of Dimension instances or 2-tuples
		# in the form (name, size)
		for d in dimensions:			

			if type(d) == tuple:
				self._dimensions.append(Dimension(*d))

			elif type(d) == Dimension:
				self._dimensions.append(d)

			else:
				raise TypeError('{} is not a 2-tuple (name, size) or a Dimension instance'.format(d))

		# Dimensions are immutable so turn the list into a tuple
		self._dimensions = tuple(self._dimensions)

		# Passed data must be an array instance
		if isinstance(data, Array):
			self._data = data
		else:
			self._data = storage(self.shape, dtype)

		self._attributes = {}

		if type(attributes) == dict:
			self._attributes = attributes
		else:
			return TypeError('attributes must be a dictionary')

	def __repr__(self):
		if self.name:
			return "<{}: {} {}>".format(self.__class__.__name__, self.name, [(d.name, d.size) for d in self.dimensions])
		else:
			return "<{}: {}>".format(self.__class__.__name__, [(d.name, d.size) for d in self.dimensions])


	def ndarray(self):
		"""
		Return variable data as an numpy instance
		"""
		return self._data.ndarray()
	
	def asjson(self, data=False):
		return {'dimensions':[], 'dtype':repr(self.dtype), 'attributes':self.attributes.copy()}


	@property
	def shape(self):

		if self._data:
			return self._data.shape
		else:
			return tuple([d.size for d in self._dimensions])

	@property
	def dtype(self):
		return self._dtype

	@property
	def dimensions(self):
		return tuple(self._dimensions)

	@property
	def attributes(self):
		return self._attributes

	def __setitem__(self, slices, values):
		self._data[slices] = values

	def __getitem__(self, slices):

		data = self._data[slices]

		dims = []
		for d, size in zip(self.dimensions, data.shape):
			dims.append(Dimension(d.name, size, d.fixed))

		result = self.__class__(dims, data.dtype, name=self.name, attributes=self.attributes, dataset=self.dataset, data=data)

		return result

