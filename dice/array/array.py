from abc import ABCMeta, abstractmethod
import copy
import numpy as np

from util import real_slices, reslice



class Array(object):
	"""
	An Array represents a multi-dimensional array.  This is meta class so that unary functions are not
	implemented.  These functions must be implemented by actual implementations in derived classes
	"""

	__metaclass__ = ABCMeta

	def __init__(self, shape, dtype):

		if type(shape) == tuple:
			self._shape = shape
		else:
			raise TypeError('shape must be a tuple')

		self.dtype = dtype

		# Arrays may share a data store but apply different views or subsets
		self._view = real_slices(shape)


	@property
	def shape(self):

		result = []
		for s in self._view:
			if isinstance(s, slice):
				result.append(s.stop - s.start)
			else:
				result.append(len(s))

		return tuple(result)


	def __getitem__(self, slices):

		shape, view = reslice(self._shape, self._view, slices)

		result = self.__class__(shape, self.dtype)
		result._view = view
		
		return result


	def __setitem__(self, slices, values):
		
		# Combine slices with the view to get data indices
		shape, slices = reslice(self._shape, self._view, slices)
		
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


	@classmethod
	def from_ndarray(cls, source):
		"""
		Instantiate an Array instance from an existing ndarray or MaskedArray instance

		>>> import numpy as np
		>>> array = Array.from_ndarray(np.array([[1,2,3,4],[5,6,7,8]]))
		>>> print(array.shape)
		(2, 4)
		>>> print(array.dtype)
		int64
		"""

		return cls(source.shape, source.dtype)



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





