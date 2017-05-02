import numpy as np

from array import Array, real_slices, reslice

class numpyArray(Array):
	"""
	An array class that uses numpy arrays as array storage

	>>> A = numpyArray((5,3), dtype=np.float32)
	>>> A.shape
	(5, 3)
	>>> A[:] = 42.0
	>>> A[0,0]
	[[ 42.]]
	"""

	def __init__(self, shape, dtype):

		super(numpyArray, self).__init__(shape, dtype)

		# We'll lazy allocate
		self._data = False


	def __setitem__(self, slices, values):
		""" Set values from the specified slices of the array"""
		
		# Lazy allocation
		if type(self._data) == bool:
			self._data = np.empty(self.shape, dtype=self.dtype)

		# Combine slices with the view to get data indices
		shape, slices = reslice(self.shape, self._view, slices)
		
		self._data[slices] = values


	def __getitem__(self, slices):
		"""Implements array subsetting.  The method actually returns a new numpyArray instance that 
		references the original data but has modified subset slices
		>>> a = numpyArray((3,5), dtype=np.float32)
		>>> a[:] = 42
		>>> a.shape
		(3, 5)
		>>> b = a[0:2]
		>>> b.shape
		(2, 5)
		>>> b[1,1]
		[[ 42.]]
		>>> a[1,1] = 84
		>>> b[1,1]
		[[ 84.]]

		"""

		shape, view = reslice(self.shape, self._view, slices)

		result = self.__class__(shape, self.dtype)
		
		result._data = self._data
		result._view = view
		
		return result


	def copy(self, astype=None):
		"""Return a copy of the array, defaulting to a numpy ndarray"""

		# Lazy allocation
		if type(self._data) == bool:
			self._data = np.empty(self.shape, dtype=self.dtype)

		return self._data.__getitem__(self._view)


	def __repr__(self):
		return str(self.copy())

	def apply(self, func, axis=None):
		"""
		Implements functions on the array by delegating to the underlying numpy functions
		>>> a = numpyArray((16,20), dtype=np.float32)
		>>> a[:] = np.arange(320).reshape((16,20))
		>>> a.func('mean')
		[ 159.5]
		>>> a[:,:5].func('mean', axis=0)
		[[ 150.  151.  152.  153.  154.]]

		"""

		result_shape = list(self.shape)

		if axis != None:
			result_shape[axis] = 1
		else:
			result_shape = [1]

		result = self.__class__(tuple(result_shape), dtype=self.dtype)

		if func == 'mean':
			result[:] = self._data[self._view].mean(axis=axis)

		return result



	def ufunc(self, func, other):
		"""
		Implements unary functions by delegating to the underlying numpy functions
		>>> a = numpyArray((3,5), dtype=np.float32)
		>>> b = numpyArray((3,5), dtype=np.float32)
		>>> a[:] = 42
		>>> b[:] = 42
		>>> c = a + b
		>>> c[0,0]
		[[ 84.]]
		>>> d = c - 42.0
		>>> d[0,0]
		[[ 42.]]
		>>> d = 42.0 + c
		>>> d[0,0]
		[[ 126.]]
		>>> d = 0.5 * c
		>>> d[0,0]
		[[ 42.]]
		"""

		result = numpyArray(self.shape, dtype=self.dtype)

		# If other is also an Array instance, then we need to get the data out
		if isinstance(other, Array):
			other = other.copy()

		if func == 'add':
			result._data = self._data + other
		if func == 'sub':
			result._data = self._data - other
		if func == 'mul':
			result._data = self._data * other
		if func == 'div':
			result._data = self._data - other

		return result


