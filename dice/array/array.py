from abc import ABCMeta, abstractmethod
import copy
import numpy as np

from util import real_slices, reslice


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



class Array(object):
	"""
	An Array represents a multi-dimensional array.  This is meta class so that unary functions are not
	implemented.  These functions must be implemented by actual implementations in derived classes
	"""

	__metaclass__ = ABCMeta

	def __init__(self, shape, dtype):

		if type(shape) == tuple:
			self.shape = shape
		else:
			raise TypeError('shape must be a tuple')

		self.dtype = dtype

		# Arrays may share a data store but apply different views or subsets
		self._view = real_slices(self.shape)


	def __getitem__(self, slices):

		shape, view = reslice(self.shape, self._view, slices)

		result = self.__class__(shape, self.dtype)
		result._view = view
		
		return result


	def __setitem__(self, slices, values):
		
		# Combine slices with the view to get data indices
		shape, slices = reslice(self.shape, self._view, slices)
		
		self._data[slices] = values



	def ndarray(self):
		"""
		Return an ndarray version of the array data.  
		If array has no data then an empty nparray of the correct dtype is returned
		"""

		# Lazy allocation
		if type(self._data) == bool:
			self._data = np.empty(self.shape, dtype=self.dtype)

		return self._data.__getitem__(self._view)


	def ufunc(self, func, other):
		return NotImplemented()

	def __add__(self, other):
		return self.ufunc('add', other)

	def __sub__(self, other):
		return self.ufunc('sub', other)

	def __mul__(self, other):
		return self.ufunc('mul', other)

	def __div__(self, other):
		return self.ufunc('div', other)

	__radd__ = __add__
	__rmul__ = __mul__





