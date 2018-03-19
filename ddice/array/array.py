from abc import ABCMeta, abstractmethod
import copy
import numpy as np

from util import real_slices, reslice



class Array(object):
	"""
	An Array represents a multi-dimensional array.  This is meta class so that unary functions are not
	implemented.  These functions must be implemented by actual implementations in derived classes
	"""

	#__metaclass__ = ABCMeta

	def __init__(self, shape, dtype):

		if type(shape) == tuple:
			self._shape = shape
		else:
			raise TypeError('shape must be a tuple')

		self.dtype = dtype

		# Initialize the view
		self._view = real_slices(shape)


	@property
	def shape(self):
		"""
		Calculate the shape of the array considering the current view on the data
		"""

		result = []

		# Step through views axes
		for i in range(len(self._view)):

			s = self._view[i]

			# Calculate length of a slice
			if isinstance(s, slice):

				start, stop, step = s.start, s.stop, s.step

				if start == None:
					start = 0
				if stop == None:
					stop = self._shape[i]
				if step == None:
					step = 1

				result.append((stop - start)/step)

			# or get the length of a list
			else:
				result.append(len(s))

		return tuple(result)


	def __getitem__(self, slices):
		"""
		The __getitem__ method is implemented by creating a new array that references
		the same data but has a different view.
		"""

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
		if type(self._data) == bool and self._data == False:
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





