from abc import ABCMeta, abstractmethod

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


	def __getitem__(self, slices):
		return NotImplemented()

	def __setitem__(self, slices, values):
		return NotImplemented()





