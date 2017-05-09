from abc import ABCMeta, abstractmethod

from variable.variable import Variable


class DatasetError(Exception):

	def __init__(self, msg):
		self.msg = msg

	def __repr__(self):
		return self.msg


class Dataset():
	"""
	A Dataset consists of global attributes, a set of shared dimensions, and a list variables
	"""
	
	__metaclass__ = ABCMeta
	
	def __init__(self, dimensions=(), attributes={}, variables={}):
		"""
		>>> ds = Dataset(variables={'test1':Variable((('x', 5),('y',3)), float, {'name':'test'})})
		>>> print(ds._dimensions)
		[<Dimension: x (5) >, <Dimension: y (3) >]
		>>> print(ds.variables)
		{'test1': <Variable: [<Dimension: x (5) >, <Dimension: y (3) >]>}
		>>> ds.variables['test1'][:] = 42.0
		>>> print(ds.variables['test1'][0,0])
		[[ 42.]]
		"""

		self._dimensions = []
		self._attributes = {}
		self._variables = {}

		# Process/check dimensions list/tuple		
		if type(dimensions) in (tuple, list):

			for d in dimensions:			

				if type(d) == tuple:
					self._dimensions.append(Dimension(*d))

				elif type(d) == Dimension:
					self._dimensions.append(d)

				else:
					raise TypeError('{} is not a 2-tuple (name, size) or a Dimension instance'.format(d))			

		# Process/check attributes dictionary
		if type(attributes) == dict:
			self._attributes = dict([(unicode(name), unicode(value)) for name, value in attributes.items()])
		else:
			return TypeError('attributes must be a dictionary')


		# Process/check variables list and add dimensions if needed
		for name, var in variables.items():
			if isinstance(var, Variable):

				for dimension in var._dimensions:
		
					# Get all the dimension names we already have
					mynames = [d.name for d in self._dimensions]

					# Try and find a dimension with the new name
					try:
						gotitalready = mynames.index(dimension.name)
					
					# if we fail, then we simply add the new dimensions to the dataset
					except:
						self._dimensions.append(dimension)
						self._variables[name] = var

					# Otherwise we need to check if the sizes match, if not we throw an exception, if yes, we carry on
					else:
						if dimension.size != self._dimensions[gotitalready].size:
							raise DatasetError('Cannot add variable {} with dimension {} when dimension {} already exists in dataset'
												.format(var, dimension, self._dimensions[gotitalready]))


	@property
	def attributes(self):
		return self._attributes

	@property
	def dimensions(self):
		return self._dimensions

	@property
	def variables(self):
		return self._variables
		
	def __repr__(self):
		return "<{}: {} | {}>".format(self.__class__.__name__, repr(self.dimensions), repr(self.variables))
