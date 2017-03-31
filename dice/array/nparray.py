import numpy as np

from array import Array

class numpyArray(Array):
	"""
	An array class that uses numpy arrays as array storage

	>>> A = numpyArray((5,3), dtype=np.float32)
	>>> A.shape
	(5, 3)
	>>> A[:] = 42.0
	>>> A[0,0]
	42.0
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

		self._data[slices] = values


	def __getitem__(self, slices):
		""" Get values from the specified slices of the array"""

		# Lazy allocation
		if type(self._data) == bool:
			self._data = np.empty(self.shape, dtype=self.dtype)

		return self._data[slices]