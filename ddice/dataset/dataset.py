from abc import ABCMeta, abstractmethod
import re

from ddice.variable import Variable, Dimension

import netCDF4


class DatasetError(Exception):
	"""Dataset Errors"""

class Dataset(object):
	"""
	A Dataset consists of global attributes, a set of shared dimensions, and a list variables
	"""


	def __init__(self, uri=None, dataset=None, dimensions=(), attributes={}, variables={}):
		"""
		>>> ds = Dataset(variables={'test1':Variable((('x', 5),('y',3)), float, attributes={'name':'test'})})
		>>> print(ds._dimensions)
		[<Dimension: x (5) >, <Dimension: y (3) >]
		>>> print(ds.variables.keys())
		['test1']
		>>> ds.variables['test1'][:] = 42.0
		>>> print(ds.variables['test1'][0,0])
		<Variable: [('x', 1), ('y', 1)]>
		"""

		self._dimensions = []
		self._attributes = {}
		self._variables = {}


		# If we have a dataset instance then we use that and ignore dimensions, attributes and variables
		if dataset:

			dimensions = dataset.dimensions
			attributes = dataset.attributes
			variables = dataset.variables


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
			self._attributes = dict([(name, value) for name, value in attributes.items()])
		else:
			return TypeError('attributes must be a dictionary')


		# Process/check variables list and add dimensions if needed
		for name, var in variables.items():

			# Is it a Variable instance
			if isinstance(var, Variable):

				# Check all its dimensions to see if we have them already
				for dimension in var._dimensions:

					# Get all the dimension names we already have
					mynames = [d.name for d in self._dimensions]

					# if we fail, then we simply add the new dimensions to the dataset
					if dimension.name not in mynames:
						self._dimensions.append(dimension)

					# Otherwise we need to check if the sizes match, if not we throw an exception, if yes, we carry on
					else:
						if dimension.size != self._dimensions[mynames.index(dimension.name)].size:
							raise Exception('Cannot add variable {} with dimension {} when dimension {} already exists in dataset'.format(var, dimension, self._dimensions[gotitalready]))

				# Set the variables dataset attribute
				var.dataset = self

				# Add the variable to the local dictionary
				self._variables[name] = var



	@classmethod
	def open(cls, uri):

		for subclass in Dataset.__subclasses__():

			try:
				return subclass(uri=uri)
			except:
				pass

		return DatasetError("Failed to open uri {} using any available Dataset implementation".format(uri))


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

	def asjson(self):

		result = {"dimensions":[], "variables":{}, "attributes":self.attributes.copy()}

		for dim in self.dimensions:
			result['dimensions'].append(dim.asjson())

		for name, var in self.variables.items():
			result['variables'][name] = var.asjson(data=False)

		return result


