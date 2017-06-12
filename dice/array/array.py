from abc import ABCMeta, abstractmethod
import copy
import numpy as np


def real_slices(shape, slices=()):
	"""Turns a possibly truncated mixture of slices and integer indices into a full tuple with 
	the same length as dimensionality, of valid array indices (slice, integer, or iterable)
	"""

	# If slices is not a list/tuple then make it one
	if not hasattr(slices, '__iter__'):
		slices = (slices,)

	# Setup a default set of slices based on the shape, this means that missing
	# slices will take on default values
	result = [slice(0,size) for size in list(shape)]

	# Process each dimension specified
	for i in range(len(slices)):

		s = slices[i]

		# Process actual slice instances converting Nones to values
		if isinstance(s, slice):

			start, stop, step = s.start, s.stop, s.step
			
			if start == None:
				start = 0
			if stop == None:
				stop = shape[i]
			if step == None:
				step = 1

			result[i] = slice(start, stop, step)

		# Turn single int values into slice equivalents
		elif isinstance(s, int):
			result[i] = slice(slices[i],slices[i]+1,1)

		# Iterables can be used directly
		elif hasattr(s, '__iter__'):
			result[i] = s

	return tuple(result)


def reslice(shape, slices, newslices):
	"""Takes a list of slices (or integers or iterables), an array shape, and applies new slices to them to produce 
	a list of modified slices.  Returns a tuple of (shape, slices) where shape is the newly calculated shape

	>>> reslice((10,), (), (slice(3,-2),))
	(slice(3, 7, None),)
	>>> reslice((10,), (slice(3, 7, None),), ([0,2,3],))
	(array([3, 5, 6]),)
	>>> reslice((10,10), (slice(3, 7, None),), ([0,2,3],))
	(array([3, 5, 6]), slice(0, 10, None))
	>>> reslice((10,10), (slice(None), slice(3, 7, None)), ([0,2,3],))
	(array([0, 2, 3]), slice(3, 7, 1))
	>>> reslice((10,), (), (slice(2,-2),))
	(slice(2, 7, None),)
	"""

	# First get rationalized real slices
	slices = list(real_slices(shape, slices))

	# Default shape is the original shape
	newshape = list(shape)

	# Check that newslices is an iterable, if not, make it so
	if not hasattr(newslices, '__iter__'):
		newslices = (newslices,)

	for i in range(len(newslices)):
		
		s = slices[i]			
		ns = newslices[i]

		was = None

		# Convert slice instances to arrays
		if isinstance(s, slice):
			was = slice
			
			start, stop, step = s.start, s.stop, s.step

			if not start:
				start = 0
			if not stop:
				stop = shape[i] + 1
			if not step:
				step = 1

			s = np.arange(start, stop, step)

		# Integer indices also become single element arrays
		elif isinstance(s, int):
			was = int
			s = np.array(s)

		# Iterables become arrays
		elif hasattr(s, '__iter__'):
			s = np.array(s)

		# Otherwise throw an error
		else:
			raise ArrayError('Invalid slice {}'.repr(s))

		# Reslice
		s = s[ns]

		# Set new dimension size
		if hasattr(s, '__iter__'):
			newshape[i] = len(s)
		else:
			newshape[i] = 1

		
		# Try and get back to the type of slice we had originally
		if was == slice  and isinstance(ns, slice):
			slices[i] = slice(s[0], s[-1]+1, ns.step)

		elif was == int:
			slices[i] = slice(s,s+1,1)

		elif isinstance(s, int):
			slices[i] = slice(s,s+1,1)
		
		else:
			slices[i] = s


	return tuple(newshape), tuple(slices)





class ArrayError(Exception):

	def __init__(self, error):
		self.error = error

	def __str__(self):
		return self.error


class Dimension(object):
	"""
	A dimension defines the axis of a multi-dimensional array.  It simply has a name
	and a size.  A dimension can also be marked fixed indicating 
	whether the size cannot be changed

	>>> d = Dimension('test', 10)
	>>> d = Dimension('test2', 10, fixed=False)
	>>> d.size
	10
	>>> d.size = 20
	>>> d.size
	20
	"""

	def __init__(self, name, size, fixed=True):

		self.name = name

		# We temporarily make fixed false so we can set the size
		self.fixed = False
		self.size = size

		# unlimited must be boolean
		if type(fixed) == bool:
			self.fixed = fixed
		else:
			raise TypeError('unlimited must be boolean')

	@property
	def size(self):
		return self._size

	@size.setter
	def size(self, size):
		"""Changes the size of a dimension. Raises an exception if the dimension is fixed"""

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



class Array():
	"""
	An array defines a multi-dimensional array
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
		return NotImplemented()

	def __setitem__(self, slices, values):
		return NotImplemented()

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





